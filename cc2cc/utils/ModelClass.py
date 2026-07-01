import os
from collections import deque

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
        self.local_rank = 0
        self.verbose = True
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

    def init_model(self, if_validate=False, init_train=False):
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

        if self.args.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.eval_step * 32 * self.args.cosine_T,
                eta_min=self.args.cosine_eta_min,
            )
        elif self.args.scheduler == "constant":
            self.scheduler = optim.lr_scheduler.ConstantLR(self.optimizer)
        elif self.args.scheduler == "cosine_warm":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.args.eval_step * 32,
                T_mult=2,
                eta_min=self.args.cosine_eta_min,
            )
        else:
            raise ValueError(f"Unknown scheduler {self.args.scheduler}")

        param = next(self.model.parameters())
        self.device, self.dtype = param.device, param.dtype
        for name in ("cube_type", "cube_size", "input_level"):
            setattr(self, name, getattr(self.model, name))
            self.print(f"{name}: {getattr(self, name)}")

        if self.args.if_resume:
            self.print("Resuming training from checkpoint.")
            self.start_step = self.args.load_epoch

        self.dir_checkpoint = (
            CHECKPOINTS_PATH / f"checkpoint_{self.args.save_dir}"
        ).resolve()
        self.print(f"Checkpoint directory: {self.dir_checkpoint}")

        if self.state_dict is not None:
            self.model.load_state_dict(self.state_dict, strict=True)

        if self.optimizer_state_dict is not None:
            self.optimizer.load_state_dict(self.optimizer_state_dict)

        if self.args.distributed:
            self.print(f"Using DistributedDataParallel on rank {self.local_rank}")
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.local_rank]
            )

        if init_train:
            self.init_train()

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
            state_dict = checkpoint["state_dict"]
            if "module" in next(iter(state_dict)):
                # For backward compatibility with old checkpoints
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            self.state_dict = state_dict
            self.args.model = checkpoint["model"]
            self.optimizer_state_dict = checkpoint.get("optimizer")
            self.print(f"Model loaded from {load_path} with model {self.args.model}")
        else:
            self.print("Model not found, starting from scratch.")

    def init_train(self):
        loss_dict = {
            "L1Loss": torch.nn.L1Loss,
            "MSELoss": torch.nn.MSELoss,
        }
        loss_class = loss_dict[self.args.loss_ene]
        for name in (
            "ene_loss_fun",
            "atomic_loss_fun",
            "abs_loss_fun",
            "grad_loss_fun",
        ):
            setattr(self, name, loss_class(reduction="sum").cuda(self.local_rank))

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

    def save_model(self, epoch):
        if not self.dir_checkpoint.exists():
            self.print(f"Directory {self.dir_checkpoint} not found. Created!")
            (self.dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)
        state_dict = self.model.state_dict()
        torch.save(
            {
                "state_dict": state_dict,
                "model": self.args.model,
                "optimizer": self.optimizer.state_dict(),
            },
            self.dir_checkpoint / f"{epoch}.pth",
        )

    def train(self):
        self.model.train(True)

    def eval(self):
        self.model.eval()

    def model_output(self, input_, weight):
        if self.model.before_weight:
            input_ = torch.einsum("p...,pi->p...", input_, weight)
        output = self.model(input_)
        return input_, output if self.model.before_weight else output * weight

    def loss(self, batch, if_train=True):
        input_ = batch["input"]
        weight = batch["weight"]
        if_use_cuda_event = input_.is_cuda
        tensor_to_numpy = lambda x: (
            x.detach().to("cpu", non_blocking=if_use_cuda_event)
            if x.is_cuda
            else x.numpy()
        )
        sum_target = batch["energy_target"]
        data_weight = batch["data_weight"]
        loss_multiplier = batch["loss_multiplier"]
        loss_multiplier_abs = batch["loss_multiplier_abs"]
        loss_multiplier_grad = batch["loss_multiplier_grad"]
        loss_multiplier_atomic = batch["loss_multiplier_atomic"]
        data_record = {"name": batch["name"]}

        if if_train:
            input_.requires_grad = True

        input_, output = self.model_output(input_, weight)
        if not if_train:
            output = output.detach()

        sum_output = torch.sum(output)
        tot_loss = loss_multiplier * self.ene_loss_fun(
            data_weight * sum_target, data_weight * sum_output
        )
        data_record["loss_ene"] = tensor_to_numpy(torch.abs(sum_target - sum_output))
        data_record["loss_tot_ene"] = tensor_to_numpy(tot_loss)

        if if_train:
            if self.args.if_abs:
                target = batch["output"] * weight
                abs_error = torch.abs(target - output)
                if self.args.topk_abs > 0:
                    topk_indices = torch.topk(
                        abs_error.sum(dim=1), self.args.topk_abs
                    ).indices
                    loss_ene_abs = (
                        loss_multiplier_abs
                        * self.abs_loss_fun(
                            data_weight * target[topk_indices],
                            data_weight * output[topk_indices],
                        )
                        / np.sqrt(self.args.topk_abs)
                    )
                else:
                    loss_ene_abs = (
                        loss_multiplier_abs
                        * self.abs_loss_fun(data_weight * target, data_weight * output)
                        / np.sqrt(target.shape[0])
                    )
                tot_loss += loss_ene_abs
                data_record["loss_ene_abs"] = tensor_to_numpy(torch.sum(abs_error))
                data_record["loss_tot_abs"] = tensor_to_numpy(loss_ene_abs)

            if self.args.if_grad:
                grad_cc_train = batch["grad_cc_train"]
                grad2force = batch["grad2force"]
                middle_ = torch.autograd.grad(
                    outputs=torch.sum(output),
                    inputs=input_,
                    create_graph=True,
                )[0]
                force = torch.einsum("piC,piCx->x", middle_, grad2force)

                loss_grad = loss_multiplier_grad * self.grad_loss_fun(
                    data_weight * grad_cc_train, data_weight * force
                )
                tot_loss += loss_grad
                data_record["loss_grad_record"] = tensor_to_numpy(
                    torch.sum(torch.abs(grad_cc_train - force))
                )
                data_record["loss_tot_grad"] = tensor_to_numpy(loss_grad)

        if self.args.if_atomic:
            ae_target = batch["ae_target"]
            ae_output = sum_output

            for i_system, system_atom in enumerate(batch["atomic_systems"]):
                name_atom = self.database_train.atomic_name_dict.get(system_atom)
                if name_atom is None:
                    self.print(
                        f"Warning: {system_atom} not found in atomic_name_dict, "
                        "skipping atomic energy calculation."
                    )
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

            loss_atomic = loss_multiplier_atomic * self.atomic_loss_fun(
                data_weight * ae_target, data_weight * ae_output
            )
            tot_loss += loss_atomic
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
            event = torch.cuda.Event()
            event.record()
        return tot_loss, data_record, event

    def train_model(self):
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        data_record_l = DataRecordList(len(self.database_train))
        pending_records = deque()

        for batch in self.database_train.data_gpu:
            batch = self.database_train.process_batch(batch, device=self.local_rank)
            tot_loss, data_record, event = self.loss(batch)
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

    def modified_b3lyp_potential(self, middle_cube):
        if self.cube_type in ("cube", "cube5"):
            middle_cube[:, :4, self.model.cube_middle] += self.b3lyp_weights(
                middle_cube
            )
        elif self.cube_type == "center_4":
            middle_cube[:, :4] += self.b3lyp_weights(middle_cube)
        else:
            raise ValueError(f"Unknown cube type: {self.cube_type}")
        return middle_cube

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
