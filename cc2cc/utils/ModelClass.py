"""
Generate list of model.
"""

from pathlib import Path
import datetime
import os
import numpy as np

import torch
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from cc2cc.utils.env_var import MAIN_PATH, CHECKPOINTS_PATH, EDGE_SIZE, CUBE_MIDDLE
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.DataBase import DataBase


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
        self.model_name = self.args.model
        self.load = self.args.load
        self.max_norm = self.args.max_norm

        self.dir_checkpoint = (
            CHECKPOINTS_PATH
            / f"checkpoint_{datetime.datetime.today():%Y-%m-%d-%H-%M-%S}/"
        ).resolve()
        self.state_dict = None

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

    def init_model(self, if_validate=False):
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

        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype
        self.cube_type = self.model.cube_type
        self.cube_size = self.model.cube_size
        self.input_level = self.model.input_level
        self.print(f"cube type: {self.cube_type}")
        self.print(f"cube size: {self.cube_size}")
        self.print(f"input level: {self.input_level}")

        if self.args.save_dir is not None and self.args.save_dir != "":
            self.dir_checkpoint = (
                CHECKPOINTS_PATH / f"checkpoint_{self.args.save_dir}"
            ).resolve()

        if self.state_dict is not None:
            self.model.load_state_dict(self.state_dict, strict=True)

        # if (not if_validate) and (not self.args.if_grad):
        #     # model.compile does not support Double backward which is used in grad.
        #     self.model.compile(dynamic=True, mode="max-autotune-no-cudagraphs")
        #     self.print("Model compiled with torch.compile!")

        if self.args.distributed:
            self.print(f"Using DistributedDataParallel on rank {self.local_rank}")
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.local_rank]
            )

    def print(self, msg):
        """
        Print message only on the main process.
        """
        if self.verbose:
            print(msg, flush=True)

    def load_model(self):
        """
        Load the model from the checkpoint.
        """
        load_checkpoint = Path(CHECKPOINTS_PATH / f"checkpoint_{self.load}/").resolve()
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
            self.print(f"Model loaded from {load_path} with model {self.args.model}")
        else:
            self.print("Model not found, starting from scratch.")

    def save_model(self, epoch):
        """
        Save the model to the checkpoint.
        """
        if not self.dir_checkpoint.exists():
            self.print(f"Directory {self.dir_checkpoint} not found. Created!")
            (self.dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)
        state_dict = self.model.state_dict()
        torch.save(
            {"state_dict": state_dict, "model": self.args.model},
            self.dir_checkpoint / f"{epoch}.pth",
        )

    def init_train(self):
        """
        Initialize the optimizer, scheduler, loss function and checkpoint_dir.
        """
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

        if self.args.loss_ene == "L1Loss":
            self.loss_ene = torch.nn.L1Loss(reduction="sum").cuda(self.local_rank)
            self.loss_ene_atomic = torch.nn.L1Loss(reduction="sum").cuda(
                self.local_rank
            )
            self.loss_ene_abs = torch.nn.L1Loss(reduction="sum").cuda(self.local_rank)
            self.loss_grad = torch.nn.L1Loss(reduction="sum").cuda(self.local_rank)
        elif self.args.loss_ene == "MSELoss":
            self.loss_ene = torch.nn.MSELoss(reduction="sum").cuda(self.local_rank)
            self.loss_ene_atomic = torch.nn.MSELoss(reduction="sum").cuda(
                self.local_rank
            )
            self.loss_ene_abs = torch.nn.MSELoss(reduction="sum").cuda(self.local_rank)
            self.loss_grad = torch.nn.MSELoss(reduction="sum").cuda(self.local_rank)
        else:
            raise ValueError(f"Unknown loss function {self.args.loss_ene}")

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
                return np.stack(
                    [
                        x[:, : self.input_level, 0, 0, 0],
                        x[:, : self.input_level, 0, 0, 2],
                        x[:, : self.input_level, 0, 2, 0],
                        x[:, : self.input_level, 0, 2, 2],
                        x[:, : self.input_level, 1, 1, 1],
                        x[:, : self.input_level, 2, 0, 0],
                        x[:, : self.input_level, 2, 0, 2],
                        x[:, : self.input_level, 2, 2, 0],
                        x[:, : self.input_level, 2, 2, 2],
                    ],
                    axis=-1,
                )
            if self.cube_type == "cube5":
                return np.stack(
                    [
                        x[:, : self.input_level, 0, 0, 0],
                        x[:, : self.input_level, 0, 2, 2],
                        x[:, : self.input_level, 1, 1, 1],
                        x[:, : self.input_level, 2, 2, 0],
                        x[:, : self.input_level, 2, 0, 2],
                    ],
                    axis=-1,
                )
            raise ValueError(f"Unknown cube type: {self.cube_type}")

        def process_grad2force(x):
            # tipabcx -> tipCx
            if self.cube_type == "center_4":
                return x[
                    :, : self.input_level, :, [CUBE_MIDDLE], CUBE_MIDDLE, CUBE_MIDDLE, :
                ]
            if self.cube_type == "cube":
                x = x[:, : self.input_level, :, :, :, :, :].reshape(
                    x.shape[0],
                    self.input_level,
                    x.shape[2],
                    self.cube_size,
                    x.shape[-1],
                )
                return x
            if self.cube_type == "cube9":
                return np.stack(
                    [
                        x[:, : self.input_level, :, 0, 0, 0, :],
                        x[:, : self.input_level, :, 0, 0, 2, :],
                        x[:, : self.input_level, :, 0, 2, 0, :],
                        x[:, : self.input_level, :, 0, 2, 2, :],
                        x[:, : self.input_level, :, 1, 1, 1, :],
                        x[:, : self.input_level, :, 2, 0, 0, :],
                        x[:, : self.input_level, :, 2, 0, 2, :],
                        x[:, : self.input_level, :, 2, 2, 0, :],
                        x[:, : self.input_level, :, 2, 2, 2, :],
                    ],
                    axis=-2,
                )
            if self.cube_type == "cube5":
                return np.stack(
                    [
                        x[:, : self.input_level, :, 0, 0, 0, :],
                        x[:, : self.input_level, :, 0, 2, 2, :],
                        x[:, : self.input_level, :, 1, 1, 1, :],
                        x[:, : self.input_level, :, 2, 2, 0, :],
                        x[:, : self.input_level, :, 2, 0, 2, :],
                    ],
                    axis=-2,
                )
            raise ValueError(f"Unknown cube type: {self.cube_type}")

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
        sum_target = batch["energy_target"].cuda(self.local_rank)
        data_weight = batch["data_weight"]
        loss_multiplier = batch["loss_multiplier"]
        loss_multiplier_abs = batch["loss_multiplier_abs"]
        loss_multiplier_grad = batch["loss_multiplier_grad"]
        loss_multiplier_atomic = batch["loss_multiplier_atomic"]

        if if_train:
            input_.requires_grad = True
            if self.model.before_weight:
                output = self.model(torch.einsum("p...,pi->p...", input_, weight))
            else:
                output = self.model(input_) * weight
        else:
            with torch.no_grad():
                if self.model.before_weight:
                    output = self.model(torch.einsum("p...,pi->p...", input_, weight))
                else:
                    output = self.model(input_) * weight

        tot_loss = loss_multiplier * self.loss_ene(
            data_weight * sum_target, data_weight * torch.sum(output)
        )
        loss_record = np.abs((sum_target - torch.sum(output)).item())

        if if_train:
            if self.args.if_abs:
                target = batch["output"] * weight
                loss_abs_record = torch.sum(torch.abs(target - output)).item()
                if self.args.topk_abs > 0:
                    topk_indices = torch.topk(
                        torch.abs(target - output).sum(dim=1), self.args.topk_abs
                    ).indices
                    tot_loss += loss_multiplier_abs * self.loss_ene_abs(
                        data_weight * target[topk_indices],
                        data_weight * output[topk_indices],
                    )
                else:
                    tot_loss += loss_multiplier_abs * self.loss_ene_abs(
                        data_weight * target, data_weight * output
                    )
            else:
                loss_abs_record = 0.0

            if self.args.if_grad:
                middle_ = torch.autograd.grad(
                    torch.sum(output), input_, create_graph=True
                )[0]
                grad_cc_train = batch["grad_cc_train"].cuda(self.local_rank)
                grad2force = batch["grad2force"]
                force = torch.einsum("piC,tipCx->tx", middle_, grad2force)
                tot_loss += loss_multiplier_grad * self.loss_grad(grad_cc_train, force)
                loss_grad_record = torch.sum(torch.abs(grad_cc_train - force)).item()
            else:
                loss_grad_record = 0.0
        else:
            loss_abs_record = 0.0
            loss_grad_record = 0.0

        if self.args.if_atomic:
            ae_target = batch["ae_target"].cuda(self.local_rank)
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
                atomic_batch = self.database_train.process_batch_dataset(
                    atomic_batch, device=self.local_rank
                )
                atomic_input_ = atomic_batch["input"]
                atomic_weight = atomic_batch["weight"]
                if self.model.before_weight:
                    atomic_output = torch.sum(
                        self.model(
                            torch.einsum("p...,pi->p...", atomic_input_, atomic_weight)
                        )
                    )
                else:
                    atomic_output = torch.sum(self.model(atomic_input_) * atomic_weight)
                ae_output -= batch["atomic_stoichiometry"][i_system] * atomic_output

            tot_loss += loss_multiplier_atomic * self.loss_ene_atomic(
                data_weight * ae_target, data_weight * ae_output
            )
            loss_atomic_record = torch.abs(ae_target - ae_output).item()
        else:
            loss_atomic_record = 0.0

        data_record = {
            "loss_ene": AU2KCALMOL * loss_record,
            "loss_ene_abs": AU2KCALMOL * loss_abs_record,
            "loss_ene_atomic": AU2KCALMOL * loss_atomic_record,
            "loss_grad_record": AU2KCALMOL * loss_grad_record,
            "loss_tot": AU2KCALMOL * tot_loss.item(),
            "name": batch["name"],
        }

        if if_train:
            return tot_loss, data_record
        return data_record

    def train_model(self):
        """
        Train the model, one epoch.
        """
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        data_record_l = []

        for batch in self.database_train.data_gpu:
            batch = self.database_train.process_batch(batch, device=self.local_rank)
            tot_loss, data_record = self.loss(batch)

            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            data_record_l.append(data_record)
        return data_record_l

    def eval_model(self):
        """
        Evaluate the model.
        """
        self.eval()
        self.optimizer.zero_grad(set_to_none=True)
        data_record_l = []

        for batch in self.database_eval.data_gpu:
            batch = self.database_train.process_batch(batch, device=self.local_rank)
            data_record = self.loss(batch, if_train=False)

            data_record_l.append(data_record)
        return data_record_l

    def get_b3lyp_ene(self, rho_cube):
        if self.cube_type == "center_4":
            return (
                rho_cube[:, 0] * 0.08
                + rho_cube[:, 1] * 0.19
                + rho_cube[:, 2] * 0.72
                + rho_cube[:, 3] * 0.81
            )
        elif self.cube_type == "cube":
            return (
                rho_cube[:, 0, self.model.cube_middle] * 0.08
                + rho_cube[:, 1, self.model.cube_middle] * 0.19
                + rho_cube[:, 2, self.model.cube_middle] * 0.72
                + rho_cube[:, 3, self.model.cube_middle] * 0.81
            )
        elif self.cube_type == "cube5":
            return (
                rho_cube[:, 0, self.model.cube_middle] * 0.08
                + rho_cube[:, 1, self.model.cube_middle] * 0.19
                + rho_cube[:, 2, self.model.cube_middle] * 0.72
                + rho_cube[:, 3, self.model.cube_middle] * 0.81
            )
        else:
            raise ValueError(f"Unknown cube type: {self.cube_type}")

    def modified_b3lyp_potential(self, middle_cube):
        if self.cube_type == "center_4":
            middle_cube[:, 0] += 0.08
            middle_cube[:, 1] += 0.19
            middle_cube[:, 2] += 0.72
            middle_cube[:, 3] += 0.81
        elif self.cube_type == "cube":
            middle_cube[:, 0, self.model.cube_middle] += 0.08
            middle_cube[:, 1, self.model.cube_middle] += 0.19
            middle_cube[:, 2, self.model.cube_middle] += 0.72
            middle_cube[:, 3, self.model.cube_middle] += 0.81
        elif self.cube_type == "cube5":
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
        if self.model.before_weight:
            exc_cube = self.model(torch.einsum("p...,pi->p...", input_mat, weights_mat))
        else:
            exc_cube = self.model(input_mat) * weights_mat
        exc_cube += torch.einsum("i,ij->ij", self.get_b3lyp_ene(input_mat), weights_mat)
        middle_cube = torch.autograd.grad(torch.sum(exc_cube), input_mat)[0]
        exc_cube = exc_cube.detach().cpu().numpy().squeeze(-1)
        middle_cube = middle_cube.detach().cpu().numpy()
        return exc_cube, middle_cube
