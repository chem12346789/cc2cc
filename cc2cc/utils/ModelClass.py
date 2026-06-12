"""
Generate list of model.
"""

from pathlib import Path
import os
import time

import numpy as np
import pandas as pd

import torch
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from cc2cc.utils.env_var import MAIN_PATH, CHECKPOINTS_PATH, CUBE_MIDDLE
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.DataBase import DataBase

CUBE9_INDEX = np.array(
    [
        [0, 0, 0],
        [0, 0, 2],
        [0, 2, 0],
        [0, 2, 2],
        [1, 1, 1],
        [2, 0, 0],
        [2, 0, 2],
        [2, 2, 0],
        [2, 2, 2],
    ],
    dtype=np.int64,
)

CUBE5_INDEX = np.array(
    [
        [0, 0, 0],
        [0, 2, 2],
        [1, 1, 1],
        [2, 2, 0],
        [2, 0, 2],
    ],
    dtype=np.int64,
)


class DataRecordList:
    """
    DataRecordList is a list of DataRecord, which is used to record the training and evaluation results.
    """

    def __init__(self, len_batch):
        self.data_dict = {
            "loss_ene": np.zeros(len_batch),
            "loss_ene_abs": np.zeros(len_batch),
            "loss_ene_atomic": np.zeros(len_batch),
            "loss_grad_record": np.zeros(len_batch),
            "loss_tot": np.zeros(len_batch),
            "name": ["" for _ in range(len_batch)],
        }
        self.iter = 0

    def add_data_record(self, data_record):
        for key in self.data_dict.keys():
            if key in data_record:
                if key == "name":
                    self.data_dict[key][self.iter] = data_record[key]
                else:
                    self.data_dict[key][self.iter] = (
                        AU2KCALMOL * data_record[key].item()
                    )
            else:
                raise ValueError(f"Key {key} not found in data_record.")
        self.iter += 1

    def save(self, path):
        """Save the data_dict to a csv file."""
        df = pd.DataFrame(self.data_dict)
        df.to_csv(path, index=False)

    def merge(self):
        """merge the the named data"""
        df = pd.DataFrame(self.data_dict)
        df_grouped = df.groupby("name").mean().reset_index()
        self.data_dict = df_grouped.to_dict(orient="list")
        self.iter = len(self.data_dict["name"])


class ModelClass:
    """
    Model_Class
    """

    def __init__(self, args):
        """
        input:
        output:
            model_dict: dictionary of models
        """
        self.args = args
        self.start_step = 0
        self.dir_checkpoint = None
        self.state_dict = None
        self.optimizer_state_dict = None
        self._timer_stats = {}
        self._memory_stats = {}

        # for distributed training
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
            self.verbose = True

    def init_model(self, if_validate=False, init_train=False):
        """
        Initialize the model.
        """
        self.load_model()

        if (MAIN_PATH / f"cc2cc/utils/model/{self.args.model}.py").exists():
            model = getattr(
                __import__(f"cc2cc.utils.model.{self.args.model}", fromlist=["Model"]),
                "Model",
            )
        else:
            raise ValueError("Unknown model")

        self.model: torch.nn.Module = model().to(self.args.device)
        if self.args.precision == "float64":
            self.model.double()

        if self.args.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "Adafactor":
            self.optimizer = optim.Adafactor(
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
        elif self.args.scheduler == "cosine_warn":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.args.eval_step * 32,
                T_mult=2,
                eta_min=self.args.cosine_eta_min,
            )
        else:
            raise ValueError(f"Unknown scheduler {self.args.scheduler}")

        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype
        self.cube_type = self.model.cube_type
        self.cube_size = self.model.cube_size
        self.input_level = self.model.input_level
        self.print(f"cube type: {self.cube_type}")
        self.print(f"cube size: {self.cube_size}")
        self.print(f"input level: {self.input_level}")

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

        # if (not if_validate) and (not self.args.if_grad):
        #     # model.compile does not support Double backward which is used in grad.
        #     self.model.compile(dynamic=True, mode="max-autotune-no-cudagraphs")
        #     self.print("Model compiled with torch.compile!")

        if self.args.distributed:
            self.print(f"Using DistributedDataParallel on rank {self.local_rank}")
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.local_rank]
            )

        if init_train:
            self.init_train()

    def print(self, msg):
        """
        Print message only on the main process.
        """
        if self.verbose:
            print(msg, flush=True)

    def _reset_timers(self):
        self._timer_stats = {}

    def _add_timer(self, name, value_ms):
        if name not in self._timer_stats:
            self._timer_stats[name] = {"sum_ms": 0.0, "count": 0}
        self._timer_stats[name]["sum_ms"] += float(value_ms)
        self._timer_stats[name]["count"] += 1

    def _print_timer_summary(self, tag="epoch"):
        if not self._timer_stats:
            return
        items = []
        for name, stat in self._timer_stats.items():
            avg = stat["sum_ms"] / max(stat["count"], 1)
            items.append(
                f"{name}: total={stat['sum_ms']:.3f} ms, avg={avg:.3f} ms, n={stat['count']}"
            )
        self.print(f"[timer][{tag}] " + " | ".join(items))

    def _reset_memory_stats(self):
        self._memory_stats = {}

    def _add_memory(self, name, value_mb):
        if name not in self._memory_stats:
            self._memory_stats[name] = {"sum_mb": 0.0, "count": 0, "max_mb": 0.0}
        self._memory_stats[name]["sum_mb"] += float(value_mb)
        self._memory_stats[name]["count"] += 1
        self._memory_stats[name]["max_mb"] = max(
            self._memory_stats[name]["max_mb"], float(value_mb)
        )

    def _record_cuda_memory(self, prefix=""):
        if self.device.type != "cuda":
            return
        allocated_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
        reserved_mb = torch.cuda.memory_reserved(self.device) / (1024**2)
        peak_allocated_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(self.device) / (1024**2)
        self._add_memory(f"{prefix}cuda_allocated_mb", allocated_mb)
        self._add_memory(f"{prefix}cuda_reserved_mb", reserved_mb)
        self._add_memory(f"{prefix}cuda_peak_allocated_mb", peak_allocated_mb)
        self._add_memory(f"{prefix}cuda_peak_reserved_mb", peak_reserved_mb)

    def _print_memory_summary(self, tag="epoch"):
        if not self._memory_stats:
            return
        items = []
        for name, stat in self._memory_stats.items():
            avg = stat["sum_mb"] / max(stat["count"], 1)
            items.append(
                f"{name}: avg={avg:.2f} MB, max={stat['max_mb']:.2f} MB, n={stat['count']}"
            )
        self.print(f"[memory][{tag}] " + " | ".join(items))

    def load_model(self):
        """
        Load the model from the checkpoint.
        """
        if self.args.load is None or self.args.load == "":
            self.print("No checkpoint specified, starting from scratch.")
            return
        load_checkpoint = Path(
            CHECKPOINTS_PATH / f"checkpoint_{self.args.load}/"
        ).resolve()
        load_path = load_checkpoint / f"{self.args.load_epoch}.pth"
        self.print(f"Checking path {load_path}")
        if load_path.exists():
            self.print("Loading model from path")
            checkpoint = torch.load(
                load_path, map_location=self.args.device, weights_only=True
            )
            state_dict = checkpoint["state_dict"]
            if "module" in list(state_dict.keys())[0]:
                # For backward compatibility with old checkpoints
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            self.state_dict = state_dict
            self.args.model = checkpoint["model"]
            if "optimizer" in checkpoint:
                self.optimizer_state_dict = checkpoint["optimizer"]
            self.print(f"Model loaded from {load_path} with model {self.args.model}")
        else:
            self.print("Model not found, starting from scratch.")

    def init_train(self):
        """
        Initialize the optimizer, scheduler, loss function and checkpoint_dir.
        """
        loss_dict = {
            "L1Loss": torch.nn.L1Loss,
            "MSELoss": torch.nn.MSELoss,
        }
        loss_class = loss_dict.get(self.args.loss_ene)
        if loss_class is None:
            raise ValueError(f"Unknown loss function {self.args.loss_ene}")
        self.loss_ene = loss_class(reduction="sum").cuda(self.local_rank)
        self.loss_ene_atomic = loss_class(reduction="sum").cuda(self.local_rank)
        self.loss_ene_abs = loss_class(reduction="sum").cuda(self.local_rank)
        self.loss_grad = loss_class(reduction="sum").cuda(self.local_rank)

    def init_database(self, train_str_dict, eval_str_dict):
        """
        Initialize the database.
        """

        def process_input(x):
            if self.cube_type == "center_4":
                return x[:, : self.input_level, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            if self.cube_type == "cube":
                return x[:, : self.input_level, :, :, :].reshape(
                    x.shape[0], self.input_level, self.cube_size
                )
            if self.cube_type == "cube9":
                return x[
                    :,
                    : self.input_level,
                    CUBE9_INDEX[:, 0],
                    CUBE9_INDEX[:, 1],
                    CUBE9_INDEX[:, 2],
                ]
            if self.cube_type == "cube5":
                return x[
                    :,
                    : self.input_level,
                    CUBE5_INDEX[:, 0],
                    CUBE5_INDEX[:, 1],
                    CUBE5_INDEX[:, 2],
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
            elif self.cube_type == "cube9":
                x = x[
                    :,
                    : self.input_level,
                    :,
                    CUBE9_INDEX[:, 0],
                    CUBE9_INDEX[:, 1],
                    CUBE9_INDEX[:, 2],
                    :,
                ]
            elif self.cube_type == "cube5":
                x = x[
                    :,
                    : self.input_level,
                    :,
                    CUBE5_INDEX[:, 0],
                    CUBE5_INDEX[:, 1],
                    CUBE5_INDEX[:, 2],
                    :,
                ]
            else:
                raise ValueError(f"Unknown cube type: {self.cube_type}")
            # flatten the last two dimensions to get piCX
            x = np.transpose(x, (2, 1, 3, 0, 4))
            shape_x = x.shape
            x = x.reshape(shape_x[0], shape_x[1], shape_x[2], -1)
            return x

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
        """
        Save the model to the checkpoint.
        """
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

    def _forward(self, input_, weight):
        """
        Run the model and return:
            model_output: direct output from the model
            weighted_output: weighted output used for energy/loss terms
        """
        if self.model.before_weight:
            model_output = self.model(torch.einsum("p...,pi->p...", input_, weight))
            weighted_output = model_output
        else:
            model_output = self.model(input_)
            weighted_output = model_output * weight
        return model_output, weighted_output

    def train(self):
        """
        Set the model to train mode.
        """
        self.model.train(True)

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.model.eval()

    def loss(self, batch, if_train=True):
        """
        Calculate the loss.
        ae if for atomic energy.
        """
        input_ = batch["input"]
        weight = batch["weight"]
        sum_target = batch["energy_target"]
        data_weight = batch["data_weight"]
        loss_multiplier = batch["loss_multiplier"]
        loss_multiplier_abs = batch["loss_multiplier_abs"]
        loss_multiplier_grad = batch["loss_multiplier_grad"]
        loss_multiplier_atomic = batch["loss_multiplier_atomic"]

        if if_train:
            input_.requires_grad = True

        t_forward_start = time.perf_counter()
        model_output, output = self._forward(input_, weight)
        self._add_timer(
            "model_forward_ms", (time.perf_counter() - t_forward_start) * 1000.0
        )

        if not if_train:
            output = output.detach()

        tot_loss = loss_multiplier * self.loss_ene(
            data_weight * sum_target, data_weight * torch.sum(output)
        )
        loss_record = torch.abs((sum_target - torch.sum(output)))

        if if_train:
            if self.args.if_abs:
                target = batch["output"] * weight
                loss_abs_record = torch.sum(torch.abs(target - output))
                if self.args.topk_abs > 0:
                    topk_indices = torch.topk(
                        torch.abs(target - output).sum(dim=1), self.args.topk_abs
                    ).indices
                    tot_loss += (
                        loss_multiplier_abs
                        * self.loss_ene_abs(
                            data_weight * target[topk_indices],
                            data_weight * output[topk_indices],
                        )
                        / np.sqrt(self.args.topk_abs)
                    )
                else:
                    tot_loss += (
                        loss_multiplier_abs
                        * self.loss_ene_abs(data_weight * target, data_weight * output)
                        / np.sqrt(target.shape[0])
                    )
            else:
                loss_abs_record = torch.zeros_like(loss_record)

            if self.args.if_grad:
                if self.model.before_weight:
                    grad_output = torch.ones_like(model_output)
                else:
                    grad_output = weight
                grad_cc_train = batch["grad_cc_train"]
                grad2force = batch["grad2force"]
                if input_.is_cuda:
                    grad_start = torch.cuda.Event(enable_timing=True)
                    grad_end = torch.cuda.Event(enable_timing=True)
                    einsum_start = torch.cuda.Event(enable_timing=True)
                    einsum_end = torch.cuda.Event(enable_timing=True)
                    grad_start.record()
                    middle_ = torch.autograd.grad(
                        outputs=model_output,
                        inputs=input_,
                        grad_outputs=grad_output,
                        create_graph=True,
                    )[0]
                    grad_end.record()
                    einsum_start.record()
                    force = torch.einsum("piC,piCx->x", middle_, grad2force)
                    einsum_end.record()
                    einsum_end.synchronize()
                    grad_time_ms = grad_start.elapsed_time(grad_end)
                    einsum_time_ms = einsum_start.elapsed_time(einsum_end)
                else:
                    t0 = time.perf_counter()
                    middle_ = torch.autograd.grad(
                        outputs=model_output,
                        inputs=input_,
                        grad_outputs=grad_output,
                        create_graph=True,
                    )[0]
                    t1 = time.perf_counter()
                    force = torch.einsum("piC,piCx->x", middle_, grad2force)
                    t2 = time.perf_counter()
                    grad_time_ms = (t1 - t0) * 1000.0
                    einsum_time_ms = (t2 - t1) * 1000.0
                self._add_timer("autograd_grad_ms", grad_time_ms)
                self._add_timer("einsum_force_ms", einsum_time_ms)

                tot_loss += loss_multiplier_grad * self.loss_grad(
                    data_weight * grad_cc_train, data_weight * force
                )
                loss_grad_record = torch.sum(torch.abs(grad_cc_train - force))
            else:
                loss_grad_record = torch.zeros_like(loss_record)
        else:
            loss_abs_record = torch.zeros_like(loss_record)
            loss_grad_record = torch.zeros_like(loss_record)

        if self.args.if_atomic:
            t_atomic_start = time.perf_counter()
            ae_target = batch["ae_target"]
            ae_output = torch.sum(output)

            for i_system in range(len(batch["atomic_systems"])):
                system_atom = batch["atomic_systems"][i_system]
                if system_atom in self.database_train.atomic_name_dict:
                    name_atom = self.database_train.atomic_name_dict[system_atom]
                else:
                    self.print(
                        f"Warning: {system_atom} not found in atomic_name_dict, "
                        "skipping atomic energy calculation."
                    )
                    break
                atomic_batch = self.database_train.dataset.get_from_name(name_atom)
                atomic_batch = self.database_train.process_batch(
                    atomic_batch, device=self.local_rank
                )
                atomic_input_ = atomic_batch["input"]
                atomic_weight = atomic_batch["weight"]
                _, atomic_weighted_output = self._forward(atomic_input_, atomic_weight)
                atomic_output = torch.sum(atomic_weighted_output)
                ae_output -= batch["atomic_stoichiometry"][i_system] * atomic_output

            tot_loss += loss_multiplier_atomic * self.loss_ene_atomic(
                data_weight * ae_target, data_weight * ae_output
            )
            loss_atomic_record = torch.abs(ae_target - ae_output)
            self._add_timer(
                "atomic_loss_ms", (time.perf_counter() - t_atomic_start) * 1000.0
            )
        else:
            loss_atomic_record = torch.zeros_like(loss_record)

        event = torch.cuda.Event()
        event.record()
        data_record = {
            "loss_ene": loss_record.detach().to("cpu", non_blocking=True),
            "loss_ene_abs": loss_abs_record.detach().to("cpu", non_blocking=True),
            "loss_ene_atomic": loss_atomic_record.detach().to("cpu", non_blocking=True),
            "loss_grad_record": loss_grad_record.detach().to("cpu", non_blocking=True),
            "loss_tot": tot_loss.detach().to("cpu", non_blocking=True),
            "name": batch["name"],
        }

        return tot_loss, data_record, event

    def train_model(self):
        """
        Train the model, one epoch.
        """
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._reset_timers()
        self._reset_memory_stats()
        data_record_l = DataRecordList(len(self.database_train))
        data_record = None

        for batch in self.database_train.data_gpu:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            t_batch_start = time.perf_counter()
            t_process_batch_start = time.perf_counter()
            batch = self.database_train.process_batch(batch, device=self.local_rank)
            self._add_timer(
                "process_batch_ms",
                (time.perf_counter() - t_process_batch_start) * 1000.0,
            )
            t_loss_start = time.perf_counter()
            tot_loss, data_record, event = self.loss(batch)
            self._add_timer(
                "loss_forward_total_ms", (time.perf_counter() - t_loss_start) * 1000.0
            )

            if tot_loss.is_cuda:
                backward_start = torch.cuda.Event(enable_timing=True)
                backward_end = torch.cuda.Event(enable_timing=True)
                backward_start.record()
                tot_loss.backward()
                backward_end.record()
                backward_end.synchronize()
                backward_time_ms = backward_start.elapsed_time(backward_end)
            else:
                t0 = time.perf_counter()
                tot_loss.backward()
                t1 = time.perf_counter()
                backward_time_ms = (t1 - t0) * 1000.0
            self._add_timer("tot_loss_backward_ms", backward_time_ms)

            t_clip_start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
            self._add_timer(
                "clip_grad_norm_ms", (time.perf_counter() - t_clip_start) * 1000.0
            )

            t_step_start = time.perf_counter()
            self.optimizer.step()
            self._add_timer(
                "optimizer_step_ms", (time.perf_counter() - t_step_start) * 1000.0
            )

            t_zero_start = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            self._add_timer(
                "zero_grad_ms", (time.perf_counter() - t_zero_start) * 1000.0
            )

            t_event_wait_start = time.perf_counter()
            while data_record is not None:
                if event.query():
                    data_record_l.add_data_record(data_record)
                    data_record = None
                else:
                    break
            self._add_timer(
                "event_query_wait_ms",
                (time.perf_counter() - t_event_wait_start) * 1000.0,
            )
            self._add_timer(
                "train_batch_total_ms", (time.perf_counter() - t_batch_start) * 1000.0
            )
            self._record_cuda_memory(prefix="train_")
        self._print_timer_summary(tag="train")
        self._print_memory_summary(tag="train")
        data_record_l.merge()
        return data_record_l

    def eval_model(self):
        """
        Evaluate the model.
        """
        self.eval()
        self.optimizer.zero_grad(set_to_none=True)
        self._reset_timers()
        self._reset_memory_stats()
        data_record_l = DataRecordList(len(self.database_eval))

        for batch in self.database_eval.data_gpu:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            t_batch_start = time.perf_counter()
            t_process_batch_start = time.perf_counter()
            batch = self.database_eval.process_batch(batch, device=self.local_rank)
            self._add_timer(
                "eval_process_batch_ms",
                (time.perf_counter() - t_process_batch_start) * 1000.0,
            )
            t_loss_start = time.perf_counter()
            _, data_record, event = self.loss(batch, if_train=False)
            self._add_timer(
                "eval_loss_forward_total_ms",
                (time.perf_counter() - t_loss_start) * 1000.0,
            )

            t_event_wait_start = time.perf_counter()
            while data_record is not None:
                if event.query():
                    data_record_l.add_data_record(data_record)
                    data_record = None
                else:
                    break
            self._add_timer(
                "eval_event_query_wait_ms",
                (time.perf_counter() - t_event_wait_start) * 1000.0,
            )
            self._add_timer(
                "eval_batch_total_ms", (time.perf_counter() - t_batch_start) * 1000.0
            )
            self._record_cuda_memory(prefix="eval_")
        self._print_timer_summary(tag="eval")
        self._print_memory_summary(tag="eval")
        data_record_l.merge()
        return data_record_l

    def get_b3lyp_ene(self, rho_cube):
        if self.cube_type == "center_4":
            return (
                rho_cube[:, 0] * 0.08
                + rho_cube[:, 1] * 0.19
                + rho_cube[:, 2] * 0.72
                + rho_cube[:, 3] * 0.81
            )
        if self.cube_type in ("cube", "cube5"):
            return (
                rho_cube[:, 0, self.model.cube_middle] * 0.08
                + rho_cube[:, 1, self.model.cube_middle] * 0.19
                + rho_cube[:, 2, self.model.cube_middle] * 0.72
                + rho_cube[:, 3, self.model.cube_middle] * 0.81
            )
        raise ValueError(f"Unknown cube type: {self.cube_type}")

    def modified_b3lyp_potential(self, middle_cube):
        if self.cube_type == "center_4":
            middle_cube[:, 0] += 0.08
            middle_cube[:, 1] += 0.19
            middle_cube[:, 2] += 0.72
            middle_cube[:, 3] += 0.81
        elif self.cube_type in ("cube", "cube5"):
            middle_cube[:, 0, self.model.cube_middle] += 0.08
            middle_cube[:, 1, self.model.cube_middle] += 0.19
            middle_cube[:, 2, self.model.cube_middle] += 0.72
            middle_cube[:, 3, self.model.cube_middle] += 0.81
        else:
            raise ValueError(f"Unknown cube type: {self.cube_type}")
        return middle_cube

    def eval_xc_eff(self, rho_cube, weights_):
        """
        Get the exc and vxc from the model, for restricted Kohn-Sham (RKS) calculations.
        Args:
            rho_cube: Electron density on the cube grid.
        Returns:
            exc: Exchange-correlation energy.
            vxc: Exchange-correlation potential.
        """
        input_mat = torch.tensor(rho_cube, dtype=self.dtype, device=self.device)
        weights_mat = torch.tensor(
            weights_.reshape((-1, 1)), dtype=self.dtype, device=self.device
        )
        input_mat.requires_grad = True
        _, exc_cube = self._forward(input_mat, weights_mat)
        exc_cube += torch.einsum("i,ij->ij", self.get_b3lyp_ene(input_mat), weights_mat)
        middle_cube = torch.autograd.grad(torch.sum(exc_cube), input_mat)[0]
        exc_cube = exc_cube.detach().cpu().numpy().squeeze(-1)
        middle_cube = middle_cube.detach().cpu().numpy()
        return exc_cube, middle_cube
