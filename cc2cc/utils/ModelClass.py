import faulthandler
import os
import time
from collections import deque
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from cc2cc.utils.DataBase import DataBase
from cc2cc.utils.helper.runtime.env_var import CHECKPOINTS_PATH, MAIN_PATH
from cc2cc.utils.helper.model.compilation import maybe_compile_model
from cc2cc.utils.helper.model.data_record import DataRecordList, TORCH_LIST
from cc2cc.utils.helper.model.data_processing import process_grad2force, process_input
from cc2cc.utils.helper.model.checkpoint import (
    load_model,
    save_model as _save_model,
)
from cc2cc.utils.helper.model.schedulers import select_scheduler
from cc2cc.utils.helper.model.optimizers import select_optimizer

B3LYP_WEIGHTS = (0.08, 0.19, 0.72, 0.81)

DEBUG = 0


@contextmanager
def training_stage(stage, batch=None):
    setting = os.environ.get("DFT2CC_TRACE_TRAIN", "0")
    if setting not in ("0", "1"):
        raise ValueError("DFT2CC_TRACE_TRAIN must be 0 or 1")
    if setting == "0":
        yield
        return

    label = f"[train-progress] pid={os.getpid()} stage={stage}"
    device = None
    if batch is not None:
        device = batch["input"].device
        label += f" name={batch.get('name', 'atomic')} device={device} grids={len(batch['input'])}"
    print(f"{label} start", flush=True)
    start = time.perf_counter()
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        yield
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        faulthandler.cancel_dump_traceback_later()
    print(f"{label} done wall_s={time.perf_counter() - start:.3f}", flush=True)


class ModelClass:
    def __init__(self, args):
        self.args = args
        self.start_step = 0
        self.dir_checkpoint = None
        self.state_dict = None
        self.optimizer_state_dict = None
        self.rng_state_dict = None
        self.rng_state_for_save = None

        # for distributed training
        self.verbose = True
        if args.device == "cuda":
            if self.args.distributed:
                self.local_rank = int(os.environ["LOCAL_RANK"])
                torch.cuda.set_device(self.local_rank)
                dist.init_process_group(
                    backend="nccl",
                    rank=self.local_rank,
                    world_size=torch.cuda.device_count(),
                    device_id=torch.device("cuda", self.local_rank),
                )
                self.verbose = dist.get_rank() == 0
            else:
                self.local_rank = 0
        else:
            self.local_rank = "cpu"

    def init_model(self, init_train=False):
        load_model(self, CHECKPOINTS_PATH)

        if not (MAIN_PATH / f"cc2cc/utils/model/{self.args.model}.py").exists():
            raise ValueError("Unknown model")
        model = getattr(
            __import__(f"cc2cc.utils.model.{self.args.model}", fromlist=["Model"]),
            "Model",
        )

        self.model: torch.nn.Module = model().to(self.args.device)
        if self.args.precision == "float64":
            self.model.double()

        param = next(self.model.parameters())
        self.device, self.dtype = param.device, param.dtype
        self.cube_type = self.model.cube_type
        self.cube_size = self.model.cube_size
        self.input_level = self.model.input_level
        for name in ("cube_type", "cube_size", "input_level"):
            self.print(f"{name}: {getattr(self, name)}")

        if self.state_dict is not None:
            self.model.load_state_dict(self.state_dict, strict=True)
            self.state_dict = None

        maybe_compile_model(self)

        if init_train:
            if self.args.if_resume:
                self.print("Resuming training from checkpoint.")
                # Checkpoints contain the updates from the named epoch.
                self.start_step = int(self.args.load_epoch) + 1

            self.dir_checkpoint = (
                CHECKPOINTS_PATH / f"checkpoint_{self.args.save_dir}"
            ).resolve()
            self.print(f"Checkpoint directory: {self.dir_checkpoint}")

            if self.args.distributed:
                self.print(f"Using DistributedDataParallel on rank {self.local_rank}")
                self.model = DistributedDataParallel(
                    self.model, device_ids=[self.local_rank]
                )

            self.init_train()

    def print(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def init_train(self):
        # args.optimizer selects AdamW, Adafactor, or Muon; args.lr and
        # args.weight_decay configure all three. Muon uses args.muon_lr for
        # matrix parameters except mixing_weight, and AdamW for the remainder.
        self.optimizer = select_optimizer(self.args, self.model)
        if self.optimizer_state_dict is not None:
            self.optimizer.load_state_dict(self.optimizer_state_dict)
            self.optimizer_state_dict = None

        self.scheduler = None

        loss_dict = {
            "L1Loss": torch.nn.L1Loss,
            "MSELoss": torch.nn.MSELoss,
        }
        loss_class = loss_dict[self.args.loss_type]
        self.loss_fun = loss_class(reduction="sum").cuda(self.local_rank)

    def init_database(self, train_str_dict, eval_str_dict):
        input_processor = partial(
            process_input,
            cube_type=self.cube_type,
            input_level=self.input_level,
            cube_size=self.cube_size,
        )
        grad2force_processor = partial(
            process_grad2force,
            cube_type=self.cube_type,
            input_level=self.input_level,
            cube_size=self.cube_size,
        )

        self.database_train = DataBase(
            train_str_dict,
            self.args,
            process_input=input_processor,
            process_grad2force=grad2force_processor,
            verbose=self.verbose,
            input_mode=self.cube_type,
        )
        self.database_eval = DataBase(
            eval_str_dict,
            self.args,
            shuffle=False,
            if_eval=True,
            atomic_name_dict=self.database_train.atomic_name_dict,
            atomic_energy_dict=self.database_train.atomic_energy_dict,
            process_input=input_processor,
            process_grad2force=grad2force_processor,
            verbose=self.verbose,
            input_mode=self.cube_type,
        )

        self.print(f"Training on {len(self.database_train)} systems.")
        self.print(f"Evaluating on {len(self.database_eval)} systems.")

        steps_per_epoch = len(self.database_train.data_gpu)
        start_update = self.start_step * steps_per_epoch
        max_lrs = [
            self.args.muon_lr if group.get("use_muon", False) else self.args.lr
            for group in self.optimizer.param_groups
        ]
        for group, max_lr in zip(self.optimizer.param_groups, max_lrs):
            group["initial_lr"] = max_lr
            group["lr"] = max_lr

        # args.scheduler selects onecycle, cosine, constant, or cosine_warm;
        # args.epoch and args.cosine_* configure their durations,
        # minimum learning rate, and restarts. Epoch counts become optimizer
        # updates, including the resume offset; max_lrs supplies each group's LR.
        self.scheduler = select_scheduler(
            self.args,
            self.optimizer,
            steps_per_epoch,
            start_update,
            max_lrs,
        )

    def save_model(self, epoch):
        return _save_model(self, epoch)

    def model_output(self, input_, weight):
        model = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )
        if model.before_weight:
            input_ = torch.einsum("p...,pi->p...", input_, weight)
        output = self.model(input_)
        return input_, output if model.before_weight else output * weight

    def loss(self, batch, if_train=True, if_grad=True):
        input_ = batch["input"]
        weight = batch["weight"]
        if_use_cuda_event = input_.is_cuda
        tensor_to_numpy = lambda x: (
            x.detach().to("cpu", non_blocking=if_use_cuda_event)
            if x.is_cuda
            else x.detach().numpy()
        )
        sum_target = batch["energy_target"]
        scale = batch["scale"]
        scale_abs = batch["scale_abs"]
        scale_grad = batch["scale_grad"]
        scale_atomic = batch["scale_atomic"]
        data_record = {"name": batch["name"]}

        if DEBUG:
            print(f"{batch['name']}")

        if if_train:
            input_.requires_grad_(
                self.args.if_grad and if_grad and batch["grad_cc_train"].ndim != 0
            )
            weight.requires_grad_(input_.requires_grad)
        else:
            input_ = input_.detach()

        density_input = input_
        with training_stage("forward" if if_train else "eval-forward", batch):
            input_, output = self.model_output(input_, weight)
        if not if_train:
            output = output.detach()

        sum_output = torch.sum(output)
        if if_train:
            tot_loss = scale * self.loss_fun(sum_target, sum_output)
            data_record["loss_tot_ene"] = tensor_to_numpy(tot_loss)
        else:
            tot_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        data_record["loss_ene"] = tensor_to_numpy(torch.abs(sum_target - sum_output))

        if if_train:
            if self.args.if_abs:
                target = batch["output"] * weight
                abs_error = torch.abs(target - output)
                loss_ene_abs = scale_abs * self.loss_fun(target, output)
                tot_loss = tot_loss + loss_ene_abs
                data_record["loss_ene_abs"] = tensor_to_numpy(torch.sum(abs_error))
                data_record["loss_tot_abs"] = tensor_to_numpy(loss_ene_abs)

            if self.args.if_grad and if_grad:
                grad_cc_train = batch["grad_cc_train"]
                grad2force = batch["grad2force"]
                if len(grad_cc_train.shape) != 0:
                    middle_, weight_derivative = torch.autograd.grad(
                        outputs=torch.sum(output),
                        inputs=(density_input, weight),
                        create_graph=True,
                    )
                    force = torch.einsum("piC,piCx->x", middle_, grad2force)
                    force = force + torch.einsum(
                        "p,px->x",
                        weight_derivative.reshape(-1),
                        batch["grid_weight_response"],
                    )

                    loss_grad = scale_grad * self.loss_fun(grad_cc_train, force)
                    tot_loss = tot_loss + loss_grad
                    data_record["loss_grad_record"] = tensor_to_numpy(
                        torch.sum(torch.abs(grad_cc_train - force))
                    )
                    data_record["loss_tot_grad"] = tensor_to_numpy(loss_grad)

        if self.args.if_atomic and scale_atomic != 0:
            ae_target = batch["ae_target"]
            ae_output = sum_output.clone()

            if_finish = True
            for i_system, system_atom in enumerate(batch["atomic_systems"]):
                name_atom = self.database_train.atomic_name_dict.get(system_atom)
                if DEBUG:
                    print(
                        f"atom {system_atom} and stoichiometry {batch['atomic_stoichiometry'][i_system]}"
                    )
                if name_atom is None:
                    self.print(
                        f"Warning: {system_atom} not found in atomic_name_dict, "
                        "skipping atomic energy calculation."
                    )
                    if_finish = False
                    break
                atomic_batch = self.database_train.dataset.get_from_name(name_atom)
                atomic_batch = self.database_train.process_batch(
                    {
                        key: atomic_batch[key]
                        for key in ("input", "weight")
                        if key in atomic_batch
                    },
                    device=self.local_rank,
                )
                with training_stage(f"atomic-forward:{name_atom}", atomic_batch):
                    _, atomic_output = self.model_output(
                        atomic_batch["input"],
                        atomic_batch["weight"],
                    )
                atomic_output = torch.sum(atomic_output)
                ae_output -= batch["atomic_stoichiometry"][i_system] * atomic_output

            if if_finish:
                if if_train:
                    loss_atomic = scale_atomic * self.loss_fun(ae_target, ae_output)
                    tot_loss = tot_loss + loss_atomic
                    data_record["loss_tot_atomic"] = tensor_to_numpy(loss_atomic)
                data_record["loss_ene_atomic"] = tensor_to_numpy(
                    torch.abs(ae_target - ae_output)
                )

        data_record["loss_tot"] = tensor_to_numpy(tot_loss)
        for key in TORCH_LIST:
            if key not in data_record:
                data_record[key] = np.array(0.0)

        event = None
        if if_use_cuda_event:
            final_event = torch.cuda.Event()
            final_event.record()
            event = final_event

        if DEBUG:
            print(data_record)
            print("end of loss\n")
        return tot_loss, data_record, event

    def train_model(self, if_grad=True):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        data_record_l = DataRecordList(len(self.database_train))
        pending_records = deque()

        batches = iter(self.database_train.data_gpu)
        for step in range(len(self.database_train.data_gpu)):
            with training_stage(f"train-fetch:{step}"):
                batch = next(batches)
            batch = self.database_train.process_batch(batch, device=self.local_rank)
            tot_loss, data_record, event = self.loss(batch, if_grad=if_grad)
            with training_stage("backward", batch):
                tot_loss.backward()
            with training_stage("optimizer", batch):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            if event is None:
                data_record_l.add_data_record(data_record)
                continue

            pending_records.append((event, data_record))
            while pending_records and pending_records[0][0].query():
                _, ready_record = pending_records.popleft()
                data_record_l.add_data_record(ready_record)

        while pending_records:
            event, ready_record = pending_records.popleft()
            event.synchronize()
            data_record_l.add_data_record(ready_record)
        data_record_l.trim()
        data_record_l.merge()
        return data_record_l

    @torch.inference_mode()
    def eval_model(self):
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        data_record_l = DataRecordList(len(self.database_eval))
        pending_records = deque()

        batches = iter(self.database_eval.data_gpu)
        for step in range(len(self.database_eval.data_gpu)):
            with training_stage(f"eval-fetch:{step}"):
                batch = next(batches)
            batch = self.database_eval.process_batch(batch, device=self.local_rank)
            _, data_record, event = self.loss(batch, if_train=False)
            if event is None:
                data_record_l.add_data_record(data_record)
                continue

            pending_records.append((event, data_record))
            while pending_records and pending_records[0][0].query():
                _, ready_record = pending_records.popleft()
                data_record_l.add_data_record(ready_record)

        while pending_records:
            event, ready_record = pending_records.popleft()
            event.synchronize()
            data_record_l.add_data_record(ready_record)
        data_record_l.trim()
        data_record_l.merge()
        return data_record_l

    @staticmethod
    def b3lyp_weights(like):
        if torch.is_tensor(like):
            return like.new_tensor(B3LYP_WEIGHTS)
        return np.asarray(B3LYP_WEIGHTS, dtype=like.dtype)

    def get_b3lyp_ene(self, rho_cube):
        if self.cube_type == "center_4":
            rho = rho_cube[:, :4]
        elif self.cube_type in ("cube", "cube5"):
            rho = rho_cube[:, :4, self.model.cube_middle]
        else:
            raise ValueError(f"Unknown cube type: {self.cube_type}")
        return (rho * self.b3lyp_weights(rho)).sum(-1)

    def eval_xc_eff(
        self,
        rho_cube,
        weights_,
        weight_response=False,
        *,
        return_tensors=False,
    ):
        """Optionally return weight derivatives and detached on-device tensors."""
        input_mat = torch.as_tensor(rho_cube, dtype=self.dtype, device=self.device)
        weights_mat = torch.as_tensor(
            weights_.reshape((-1, 1)), dtype=self.dtype, device=self.device
        )
        input_mat.requires_grad = True
        density_input = input_mat
        weights_mat.requires_grad = weight_response
        input_mat, exc_cube = self.model_output(input_mat, weights_mat)
        baseline = self.get_b3lyp_ene(input_mat)
        exc_cube += torch.einsum("i,ij->ij", baseline, weights_mat)
        if weight_response:
            middle_cube, weight_derivative = torch.autograd.grad(
                torch.sum(exc_cube), (density_input, weights_mat)
            )
        else:
            middle_cube = torch.autograd.grad(torch.sum(exc_cube), density_input)[0]
            weight_derivative = None
        outputs = (exc_cube.detach().squeeze(-1), middle_cube.detach())
        if weight_response:
            outputs += (weight_derivative.detach().reshape(-1),)
        if return_tensors:
            return outputs
        return tuple(output.cpu().numpy() for output in outputs)
