import os
import pickle
import shutil
import time
from collections import deque
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel

from cc2cc.utils.DataBase import DataBase
from cc2cc.utils.env_var import CHECKPOINTS_PATH, CUBE_MIDDLE, MAIN_PATH
from cc2cc.utils.mol import AU2KCALMOL

CUBE_INDEX = {
    "cube5": np.array(
        [(0, 0, 0), (0, 2, 2), (1, 1, 1), (2, 2, 0), (2, 0, 2)],
        dtype=np.int64,
    ),
    "cube9": np.array(
        [
            (0, 0, 0),
            (0, 0, 2),
            (0, 2, 0),
            (0, 2, 2),
            (1, 1, 1),
            (2, 0, 0),
            (2, 0, 2),
            (2, 2, 0),
            (2, 2, 2),
        ],
        dtype=np.int64,
    ),
}
B3LYP_WEIGHTS = (0.08, 0.19, 0.72, 0.81)
TORCH_LIST = [
    "loss_ene",
    "loss_ene_abs",
    "loss_ene_atomic",
    "loss_grad_record",
    "loss_tot",
    "loss_tot_ene",
    "loss_tot_abs",
    "loss_tot_grad",
    "loss_tot_atomic",
]

STATE_DICT_PREFIXES = ("_orig_mod.",)
DEBUG = 0


class DataRecordList:
    def __init__(self, len_batch):
        self.data_dict = {
            "name": ["" for _ in range(len_batch)],
        }
        for key in TORCH_LIST:
            self.data_dict[key] = np.zeros(len_batch)
        self.iter = 0

    def add_data_record(self, data_record):
        for key, values in self.data_dict.items():
            if key not in data_record:
                raise ValueError(f"Key {key} not found in data_record.")
            value = data_record[key]
            values[self.iter] = value if key == "name" else AU2KCALMOL * value.item()
        self.iter += 1

    def save(self, path):
        pd.DataFrame(self.data_dict).to_csv(path, index=False)

    def merge(self):
        self.data_dict = (
            pd.DataFrame(self.data_dict)
            .groupby("name")
            .mean()
            .reset_index()
            .to_dict(orient="list")
        )
        self.iter = len(self.data_dict["name"])


class ModelClass:
    def __init__(self, args):
        self.args = args
        self.start_step = 0
        self.dir_checkpoint = None
        self.state_dict = None
        self.optimizer_state_dict = None

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
        self.load_model()

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

        self._maybe_compile_model()

        if init_train:
            if self.args.if_resume:
                self.print("Resuming training from checkpoint.")
                self.start_step = self.args.load_epoch

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

    def _maybe_compile_model(self):
        if self.args.if_grad:
            return

        try:
            compiled_model = torch.compile(self.model)
            self.model = cast(torch.nn.Module, compiled_model)
            self.print("Model compiled with torch.compile().")
        except Exception as exc:  # pragma: no cover - fallback path
            self.print(f"torch.compile failed ({exc}). Using eager mode.")

    def _normalize_state_dict(self, state_dict: dict[str, torch.Tensor]):
        normalized_state_dict = {}
        for key, value in state_dict.items():
            normalized_key = key
            prefix_removed = True
            while prefix_removed:
                prefix_removed = False
                for prefix in STATE_DICT_PREFIXES:
                    if normalized_key.startswith(prefix):
                        normalized_key = normalized_key[len(prefix) :]
                        prefix_removed = True
            normalized_state_dict[normalized_key] = value
        return normalized_state_dict

    def _state_dict_for_save(self):
        return self._normalize_state_dict(self.model.state_dict())

    def print(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def load_model(self):
        if not self.args.load:
            self.print("No checkpoint specified, starting from scratch.")
            return
        load_checkpoint = (CHECKPOINTS_PATH / f"checkpoint_{self.args.load}").resolve()
        load_path = load_checkpoint / f"{self.args.load_epoch}.pth"
        self.print(f"Checking path {load_path}")
        if load_path.exists():
            self.print("Loading model from path")
            checkpoint = torch.load(
                load_path, map_location=self.args.device, weights_only=True
            )
            state_dict = self._normalize_state_dict(checkpoint["state_dict"])
            self.state_dict = state_dict
            self.args.model = checkpoint["model"]
            self.optimizer_state_dict = checkpoint.get("optimizer")
            self.print(f"Model loaded from {load_path} with model {self.args.model}")
        else:
            self.print("Model not found, starting from scratch.")

    def init_train(self):
        optimizer_cls = {
            "AdamW": optim.AdamW,
            "Adafactor": getattr(optim, "Adafactor", None),
        }.get(self.args.optimizer)
        if optimizer_cls is None:
            raise ValueError(f"Unknown optimizer {self.args.optimizer}")
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        if self.optimizer_state_dict is not None:
            self.optimizer.load_state_dict(self.optimizer_state_dict)

        if self.args.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.cosine_T,
                eta_min=self.args.cosine_eta_min,
            )
        elif self.args.scheduler == "constant":
            self.scheduler = optim.lr_scheduler.ConstantLR(self.optimizer)
        elif self.args.scheduler == "cosine_warm":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.args.cosine_T,
                T_mult=self.args.cosine_T_mult,
                eta_min=self.args.cosine_eta_min,
            )
        else:
            raise ValueError(f"Unknown scheduler {self.args.scheduler}")

        loss_dict = {
            "L1Loss": torch.nn.L1Loss,
            "MSELoss": torch.nn.MSELoss,
        }
        loss_class = loss_dict[self.args.loss_type]
        self.loss_fun = loss_class(reduction="sum").cuda(self.local_rank)

    def init_database(self, train_str_dict, eval_str_dict):
        def process_input(x):
            if self.cube_type == "center_4":
                return x[:, : self.input_level, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            if self.cube_type == "cube":
                return x[:, : self.input_level, :, :, :].reshape(
                    x.shape[0], self.input_level, self.cube_size
                )
            if self.cube_type in CUBE_INDEX:
                idx = CUBE_INDEX[self.cube_type]
                return x[
                    :,
                    : self.input_level,
                    idx[:, 0],
                    idx[:, 1],
                    idx[:, 2],
                ]
            raise ValueError(f"Unknown cube type: {self.cube_type}")

        def process_grad2force(x):
            # tipabcx -> tipCx -> piCtx
            if self.cube_type == "center_4":
                x = x[
                    :, : self.input_level, :, [CUBE_MIDDLE], CUBE_MIDDLE, CUBE_MIDDLE, :
                ]
            elif self.cube_type == "cube":
                x = x[:, : self.input_level, :, :, :, :, :].reshape(
                    x.shape[0],
                    self.input_level,
                    x.shape[2],
                    self.cube_size,
                    x.shape[-1],
                )
            elif self.cube_type in CUBE_INDEX:
                idx = CUBE_INDEX[self.cube_type]
                x = x[
                    :,
                    : self.input_level,
                    :,
                    idx[:, 0],
                    idx[:, 1],
                    idx[:, 2],
                    :,
                ]
            else:
                raise ValueError(f"Unknown cube type: {self.cube_type}")
            x = np.transpose(x, (2, 1, 3, 0, 4))
            shape_x = x.shape
            return x.reshape(shape_x[0], shape_x[1], shape_x[2], -1)

        self.database_train = DataBase(
            train_str_dict,
            self.args,
            process_input=process_input,
            process_grad2force=process_grad2force,
            verbose=self.verbose,
        )
        self.database_eval = DataBase(
            eval_str_dict,
            self.args,
            shuffle=False,
            if_eval=True,
            atomic_name_dict=self.database_train.atomic_name_dict,
            atomic_energy_dict=self.database_train.atomic_energy_dict,
            process_input=process_input,
            process_grad2force=process_grad2force,
            verbose=self.verbose,
        )

        self.print(f"Training on {len(self.database_train)} systems.")
        self.print(f"Evaluating on {len(self.database_eval)} systems.")

    def _estimate_checkpoint_size(self, state_dict):
        payload = {
            "state_dict": state_dict,
            "model": self.args.model,
            "optimizer": self.optimizer.state_dict(),
        }
        return max(
            len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)), 1 << 20
        )

    def save_model(self, epoch):
        if not self.dir_checkpoint.exists():
            self.print(f"Directory {self.dir_checkpoint} not found. Created!")
            (self.dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)

        state_dict = self._state_dict_for_save()
        checkpoint_path = self.dir_checkpoint / f"{epoch}.pth"
        payload = {
            "state_dict": state_dict,
            "model": self.args.model,
            "optimizer": self.optimizer.state_dict(),
        }
        required_bytes = self._estimate_checkpoint_size(state_dict)

        while True:
            free_bytes = shutil.disk_usage(self.dir_checkpoint).free
            if free_bytes >= required_bytes:
                try:
                    torch.save(payload, checkpoint_path)
                    return
                except OSError as exc:
                    msg = str(exc).lower()
                    if "no space left" in msg or "file write failed" in msg:
                        self.print(
                            "Disk space became insufficient while writing checkpoint; "
                            f"waiting for free disk before retrying {checkpoint_path}."
                        )
                        time.sleep(30)
                        continue
                    raise

            self.print(
                "Waiting for free disk space before saving checkpoint: "
                f"{free_bytes} bytes free, {required_bytes} bytes required."
            )
            time.sleep(30 * 60)  # Wait for 30 minutes before checking again

    def train(self):
        self.model.train(True)

    def eval(self):
        self.model.eval()

    def model_output(self, input_, weight):
        if self.model.before_weight:
            input_ = torch.einsum("p...,pi->p...", input_, weight)
        output = self.model(input_)
        return input_, output if self.model.before_weight else output * weight

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
            input_.requires_grad = True
        else:
            input_ = input_.detach()

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
                    middle_ = torch.autograd.grad(
                        outputs=torch.sum(output),
                        inputs=input_,
                        create_graph=True,
                    )[0]
                    force = torch.einsum("piC,piCx->x", middle_, grad2force)

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
                    atomic_batch, device=self.local_rank
                )
                _, atomic_output = self.model_output(
                    atomic_batch["input"], atomic_batch["weight"]
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
            else:
                data_record["loss_tot_atomic"] = np.array(0.0)
                data_record["loss_ene_atomic"] = np.array(0.0)

        data_record["loss_tot"] = tensor_to_numpy(tot_loss)
        for key in TORCH_LIST:
            if key not in data_record:
                data_record[key] = np.array(0.0)

        event = None
        if if_use_cuda_event:
            event = torch.cuda.Event()
            event.record()

        if DEBUG:
            print(data_record)
            print("end of loss\n")
        return tot_loss, data_record, event

    def train_model(self, if_grad=True):
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        data_record_l = DataRecordList(len(self.database_train))
        pending_records = deque()

        for batch in self.database_train.data_gpu:
            batch = self.database_train.process_batch(batch, device=self.local_rank)
            tot_loss, data_record, event = self.loss(batch, if_grad=if_grad)
            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
            self.optimizer.step()
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
        data_record_l.merge()
        return data_record_l

    def eval_model(self):
        self.eval()
        self.optimizer.zero_grad(set_to_none=True)
        data_record_l = DataRecordList(len(self.database_eval))
        pending_records = deque()

        for batch in self.database_eval.data_gpu:
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

    def eval_xc_eff(self, rho_cube, weights_):
        input_mat = torch.as_tensor(rho_cube, dtype=self.dtype, device=self.device)
        weights_mat = torch.as_tensor(
            weights_.reshape((-1, 1)), dtype=self.dtype, device=self.device
        )
        input_mat.requires_grad = True
        input_mat, exc_cube = self.model_output(input_mat, weights_mat)
        exc_cube += torch.einsum("i,ij->ij", self.get_b3lyp_ene(input_mat), weights_mat)
        middle_cube = torch.autograd.grad(torch.sum(exc_cube), input_mat)[0]
        exc_cube = exc_cube.detach().cpu().numpy().squeeze(-1)
        middle_cube = middle_cube.detach().cpu().numpy()
        return exc_cube, middle_cube
