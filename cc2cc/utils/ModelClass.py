import os
from collections import deque
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

    def loss_ene(self, batch, sum_output):
        target = batch["energy_target"]
        return (
            batch["scale"] * self.loss_fun(target, sum_output),
            torch.abs(target - sum_output),
        )

    def loss_abs(self, batch, output):
        target = batch["output"] * batch["weight"]
        return (
            batch["scale_abs"] * self.loss_fun(target, output),
            torch.sum(torch.abs(target - output)),
        )

    def loss_grad(self, batch, sum_output):
        density_derivative, weight_derivative = torch.autograd.grad(
            outputs=sum_output,
            inputs=(batch["input"], batch["weight"]),
            create_graph=True,
        )
        force = torch.einsum("piC,piCx->x", density_derivative, batch["grad2force"])
        force = force + torch.einsum(
            "p,px->x",
            weight_derivative.reshape(-1),
            batch["grid_weight_response"],
        )
        target = batch["grad_cc_train"]
        return (
            batch["scale_grad"] * self.loss_fun(target, force),
            torch.sum(torch.abs(target - force)),
        )

    def loss_zmp_response(self, batch, sum_output):
        input_derivative = torch.autograd.grad(
            outputs=sum_output,
            inputs=batch["input"],
            create_graph=True,
        )[0].to(device=self.device, dtype=torch.float64)
        if input_derivative.ndim == 2:
            input_derivative = input_derivative.unsqueeze(2)

        prediction = batch["zmp_response_reference"].clone()
        ncenter = input_derivative.shape[0]
        block_size = 2048
        model_coefficients = []
        spin_responses = []
        spin = 0
        while f"zmp_response_features_s{spin}" in batch:
            features = batch[f"zmp_response_features_s{spin}"]
            model_coeff = torch.zeros_like(
                batch[f"zmp_response_baseline_s{spin}"],
                dtype=torch.float64,
                device=self.device,
            )
            for start in range(0, ncenter, block_size):
                stop = min(start + block_size, ncenter)
                feature_block = torch.as_tensor(
                    np.array(features[start:stop], dtype=np.float64, copy=True),
                    dtype=torch.float64,
                    device=self.device,
                )
                model_coeff = model_coeff + torch.einsum(
                    "pfc,pfck->k",
                    input_derivative[start:stop],
                    feature_block,
                )
            model_coefficients.append(model_coeff)
            spin_responses.append(
                (
                    batch[f"zmp_response_matrix_s{spin}"],
                    batch[f"zmp_response_baseline_s{spin}"],
                )
            )
            spin += 1

        if not model_coefficients:
            raise ValueError("Missing ZMP response feature sidecars in the batch.")
        model_coeff = torch.stack(model_coefficients).mean(dim=0)
        for response, baseline in spin_responses:
            prediction = prediction + torch.mv(response, model_coeff + baseline)

        target = batch["zmp_response_target"]
        return (
            batch["scale_zmp_response"] * self.loss_fun(target, prediction),
            torch.sum(torch.abs(target - prediction)),
        )

    def loss_atomic(self, batch, sum_output, if_train=True):
        ae_output = sum_output.clone()
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
                return None
            atomic_batch = self.database_train.dataset.get_from_name(name_atom)
            atomic_batch = self.database_train.process_batch(
                {
                    key: atomic_batch[key]
                    for key in ("input", "weight")
                    if key in atomic_batch
                },
                device=self.local_rank,
            )
            _, atomic_output = self.model_output(
                atomic_batch["input"], atomic_batch["weight"]
            )
            ae_output -= batch["atomic_stoichiometry"][i_system] * torch.sum(
                atomic_output
            )

        target = batch["ae_target"]
        loss = (
            batch["scale_atomic"] * self.loss_fun(target, ae_output)
            if if_train
            else None
        )
        return loss, torch.abs(target - ae_output)

    def loss(self, batch, if_train=True):
        if_use_cuda_event = batch["input"].is_cuda
        tensor_to_numpy = lambda x: (
            x.detach().to("cpu", non_blocking=if_use_cuda_event)
            if x.is_cuda
            else x.detach().numpy()
        )
        data_record = {"name": batch["name"]}

        if DEBUG:
            print(f"{batch['name']}")

        if if_train:
            batch["input"].requires_grad_(
                (
                    self.args.if_grad
                    and batch["grad_cc_train"].ndim != 0
                )
                or getattr(self.args, "if_zmp_response", False)
            )
            batch["weight"].requires_grad_(batch["input"].requires_grad)

        _, output = self.model_output(
            batch["input"] if if_train else batch["input"].detach(),
            batch["weight"],
        )
        if not if_train:
            output = output.detach()

        sum_output = torch.sum(output)
        if if_train:
            tot_loss, energy_error = self.loss_ene(batch, sum_output)
            data_record["loss_tot_ene"] = tensor_to_numpy(tot_loss)

            if self.args.if_abs:
                loss_ene_abs, abs_error = self.loss_abs(batch, output)
                tot_loss = tot_loss + loss_ene_abs
                data_record["loss_ene_abs"] = tensor_to_numpy(abs_error)
                data_record["loss_tot_abs"] = tensor_to_numpy(loss_ene_abs)

            if self.args.if_grad and batch["grad_cc_train"].ndim != 0:
                loss_grad, grad_error = self.loss_grad(batch, sum_output)
                tot_loss = tot_loss + loss_grad
                data_record["loss_grad_record"] = tensor_to_numpy(grad_error)
                data_record["loss_tot_grad"] = tensor_to_numpy(loss_grad)
            if getattr(self.args, "if_zmp_response", False):
                loss_zmp_response, zmp_response_error = self.loss_zmp_response(
                    batch, sum_output
                )
                tot_loss = tot_loss + loss_zmp_response
                data_record["loss_zmp_response"] = tensor_to_numpy(
                    zmp_response_error
                )
                data_record["loss_tot_zmp_response"] = tensor_to_numpy(
                    loss_zmp_response
                )
        else:
            tot_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            energy_error = torch.abs(batch["energy_target"] - sum_output)

        data_record["loss_ene"] = tensor_to_numpy(energy_error)

        if self.args.if_atomic and batch["scale_atomic"] != 0:
            atomic_result = self.loss_atomic(batch, sum_output, if_train=if_train)
            if atomic_result is not None:
                loss_atomic, atomic_error = atomic_result
                if if_train:
                    tot_loss = tot_loss + loss_atomic
                    data_record["loss_tot_atomic"] = tensor_to_numpy(loss_atomic)
                data_record["loss_ene_atomic"] = tensor_to_numpy(atomic_error)

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

    def train_model(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        data_record_l = DataRecordList(len(self.database_train))
        pending_records = deque()

        batches = iter(self.database_train.data_gpu)
        for _ in range(len(self.database_train.data_gpu)):
            batch = next(batches)
            batch = self.database_train.process_batch(batch, device=self.local_rank)
            tot_loss, data_record, event = self.loss(batch)
            tot_loss.backward()
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
        for _ in range(len(self.database_eval.data_gpu)):
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
