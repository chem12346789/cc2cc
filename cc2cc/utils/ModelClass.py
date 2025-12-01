"""
Generate list of model.
"""

from pathlib import Path
import datetime
import os
import numpy as np

import torch
import torch.optim as optim
import torch._functorch.config

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from cc2cc.utils.env_var import MAIN_PATH, CHECKPOINTS_PATH, CUBE_SIZE, CUBE_MIDDLE
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.DataBaseCube import DataBaseCube
from cc2cc.utils.DataBaseCenter import DataBaseCenter


class ModelClass:
    """
    Model_Class
    """

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    loss_ene: torch.nn.Module
    loss_ene_abs: torch.nn.Module
    loss_ene_atomic: torch.nn.Module
    loss_grad: torch.nn.Module
    database_train: DataBaseCube | DataBaseCenter
    database_eval: DataBaseCube | DataBaseCenter
    model: torch.nn.Module
    device: str
    dtype: str
    model_type: str

    def __init__(self, args):
        """
        input:
        output:
            model_dict: dictionary of models
        """
        self.args = args
        self.model_name = self.args.model
        self.load = self.args.load

        self.iters_to_accumulate = self.args.iters_to_accumulate
        self.max_norm = self.args.max_norm
        self.update_counter = 0

        self.dir_checkpoint = None
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
        self.model_type = self.model.model_type
        self.print(f"Model type: {self.model_type}")

        if self.args.save_dir is not None and self.args.save_dir != "":
            self.dir_checkpoint = (
                CHECKPOINTS_PATH / f"checkpoint_{self.args.save_dir}"
            ).resolve()
        else:
            self.dir_checkpoint = (
                CHECKPOINTS_PATH
                / f"checkpoint_{datetime.datetime.today():%Y-%m-%d-%H-%M-%S}/"
            ).resolve()
        if not self.dir_checkpoint.exists():
            self.print(f"Directory {self.dir_checkpoint} not found. Created!")
            (self.dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)

        if self.state_dict is not None:
            self.model.load_state_dict(self.state_dict, strict=False)
        # else:
        #     for name, param in self.model.named_parameters():
        #         if "weight" in name:
        #             self.print(f"Initialize parameter {name} with shape {param.shape}")
        #             torch.nn.init.xavier_normal_(param)
        #         elif "bias" in name:
        #             self.print(f"Initialize parameter {name} with shape {param.shape}")
        #             torch.nn.init.zeros_(param)
        #         else:
        #             self.print(f"Parameter {name} with shape {param.shape} not initialized")

        if (not if_validate) and (not self.args.if_grad):
            # model.compile does not support Double backward which is used in grad.
            self.model.compile(dynamic=True, mode="max-autotune-no-cudagraphs")
            self.print("Model compiled with torch.compile!")

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
            self.loss_ene_abs = torch.nn.L1Loss(reduction="sum").cuda(self.local_rank)
            self.loss_ene_atomic = torch.nn.L1Loss(reduction="sum").cuda(
                self.local_rank
            )
            self.loss_grad = torch.nn.L1Loss(reduction="sum").cuda(self.local_rank)
        elif self.args.loss_ene == "MSELoss":
            self.loss_ene = torch.nn.MSELoss(reduction="sum").cuda(self.local_rank)
            self.loss_ene_abs = torch.nn.MSELoss(reduction="sum").cuda(self.local_rank)
            self.loss_ene_atomic = torch.nn.MSELoss(reduction="sum").cuda(
                self.local_rank
            )
            self.loss_grad = torch.nn.MSELoss(reduction="sum").cuda(self.local_rank)
        else:
            raise ValueError(f"Unknown loss function {self.args.loss_ene}")

    def init_database(self, train_str_dict, eval_str_dict):
        """
        Initialize the database.
        """
        if self.model_type == "center_4":
            input_size = (1, 4)
            self.database_train = DataBaseCenter(
                train_str_dict,
                self.args,
                verbose=self.verbose,
            )
            self.database_eval = DataBaseCenter(
                eval_str_dict,
                self.args,
                shuffle=False,
                if_eval=True,
                atomic_name_dict=self.database_train.atomic_name_dict,
                atomic_energy_dict=self.database_train.atomic_energy_dict,
                verbose=self.verbose,
            )
        elif self.model_type == "cube":
            input_size = (1, 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
            self.database_train = DataBaseCube(
                train_str_dict,
                self.args,
                verbose=self.verbose,
            )
            self.database_eval = DataBaseCube(
                eval_str_dict,
                self.args,
                shuffle=False,
                if_eval=True,
                atomic_name_dict=self.database_train.atomic_name_dict,
                atomic_energy_dict=self.database_train.atomic_energy_dict,
                verbose=self.verbose,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        if self.local_rank == 0:
            self.print(
                f"Model {self.model_name} initialized with input size {input_size} "
                f"and model type {self.model_type}."
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
        ae_target = batch["ae_target"].cuda(self.local_rank)
        loss_multiplier = batch["loss_multiplier"]
        loss_multiplier_abs = batch["loss_multiplier_abs"]
        loss_multiplier_grad = batch["loss_multiplier_grad"]
        loss_multiplier_atomic = batch["loss_multiplier_atomic"]

        if hasattr(self.model, "normal_factor"):
            self.model.normal_factor = batch["normal_factor"]

        if if_train:
            input_.requires_grad = True
            output = self.model(input_)
            if self.args.if_grad:
                grad_cc_train = batch["grad_cc_train"].cuda(self.local_rank)
                middle_ = torch.autograd.grad(
                    torch.sum(output),
                    input_,
                    create_graph=True,
                )[0]
                if self.model_type == "cube":
                    middle_ = middle_[:, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
                grad2force = batch["grad2force"]
                force = torch.einsum("pm,impx->ix", middle_, grad2force)
            output = output * weight
        else:
            with torch.no_grad():
                output = self.model(input_) * weight

        tot_loss = loss_multiplier * self.loss_ene(
            data_weight * sum_target, data_weight * torch.sum(output)
        )
        loss_record = np.abs((sum_target - torch.sum(output)).item())

        if if_train:
            target = batch["output"] * weight
            if len(target.shape) != 0:
                # len(target.shape) will 0 if target is a dummy tensor
                tot_loss += loss_multiplier_abs * self.loss_ene_abs(
                    data_weight * target, data_weight * output
                )
            loss_abs_record = torch.sum(torch.abs(target - output)).item()

            if self.args.if_grad:
                tot_loss += loss_multiplier_grad * self.loss_grad(grad_cc_train, force)
                loss_grad_record = torch.sum(torch.abs(grad_cc_train - force)).item()
            else:
                loss_grad_record = 0.0
        else:
            loss_abs_record = 0.0
            loss_grad_record = 0.0

        if self.args.if_atomic:
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
                atomic_output = torch.sum(self.model(atomic_input_) * atomic_weight)
                ae_output -= batch["atomic_stoichiometry"][i_system] * atomic_output

            tot_loss += loss_multiplier_atomic * self.loss_ene_atomic(
                data_weight * ae_target, data_weight * ae_output
            )
            loss_atomic_record = torch.abs(ae_target - ae_output).item()
        else:
            loss_atomic_record = 0.0

        tot_loss = tot_loss / self.iters_to_accumulate
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
        1 / self.iters_to_accumulate is to match the effective batch size.
        See https://kozodoi.me/blog/20210219/gradient-accumulation and
        https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
        """
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        data_record_l = []

        for batch in self.database_train.data_gpu:
            batch = self.database_train.process_batch(batch, device=self.local_rank)
            tot_loss, data_record = self.loss(batch)

            tot_loss.backward()
            self.update_counter += 1
            if self.update_counter % self.iters_to_accumulate == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.update_counter = 0

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

    def eval_xc_eff_cube(self, rho, ni, dms, grids, coords_, mask):
        """
        Get the exc and vxc from the model, for restricted Kohn-Sham (RKS) calculations.
        Args:
            mol: PySCF molecule object.
            dms: Density matrices.
            ni: NumInt object.
            coords_: Coordinates of the grid points.
            weights_: Weights of the grid points.
        Returns:
            exc: Exchange-correlation energy.
            vxc: Exchange-correlation potential.
        """
        if grids.mol.spin == 0:
            rho_cube, exc_b3lyp, vxc_b3lyp = grids.gen_cube_rho_rks(
                rho, ni, dms, coords=coords_, mask=mask, require_vxc=True
            )
        else:
            rho_cube, exc_b3lyp, vxc_b3lyp = grids.gen_cube_rho_uks(
                rho, ni, dms, coords=coords_, mask=mask, require_vxc=True
            )

        # return exc_b3lyp, (
        #     0.08 * vxc_b3lyp[0]
        #     + 0.19 * vxc_b3lyp[1]
        #     + 0.72 * vxc_b3lyp[2]
        #     + 0.81 * vxc_b3lyp[3]
        # )

        input_mat = torch.tensor(
            rho_cube,
            dtype=self.dtype,
            device=self.device,
        )
        input_mat.requires_grad = True
        output_mat = self.model(input_mat)
        middle_cube = torch.autograd.grad(torch.sum(output_mat), input_mat)[0]
        middle_mat = grids.get_center_density(middle_cube).detach().cpu().numpy()
        energy_den = exc_b3lyp + output_mat[:, 0].detach().cpu().numpy()
        vxc = (
            (0.08 + middle_mat[:, 0]) * vxc_b3lyp[0]
            + (0.19 + middle_mat[:, 1]) * vxc_b3lyp[1]
            + (0.72 + middle_mat[:, 2]) * vxc_b3lyp[2]
            + (0.81 + middle_mat[:, 3]) * vxc_b3lyp[3]
        )
        return energy_den, vxc

    def eval_xc_eff_4(self, rho, ni, dms, grids, coords_, mask):
        """
        Get the exc and vxc from the model, for restricted Kohn-Sham (RKS) calculations.
        Args:
            mol: PySCF molecule object.
            dms: Density matrices.
            ni: NumInt object.
            coords_: Coordinates of the grid points.
            weights_: Weights of the grid points.
        Returns:
            exc: Exchange-correlation energy.
            vxc: Exchange-correlation potential.
        """
        if grids.mol.spin == 0:
            rho_b3lyp, exc_b3lyp, vxc_b3lyp = grids.gen_rho_rks(
                rho, ni, require_vxc=True
            )
        else:
            rho_b3lyp, exc_b3lyp, vxc_b3lyp = grids.gen_rho_uks(
                rho, ni, require_vxc=True
            )

        input_mat = torch.tensor(
            rho_b3lyp,
            dtype=self.dtype,
            device=self.device,
        )
        input_mat.requires_grad = True
        output_mat = self.model(input_mat)
        middle_cube = torch.autograd.grad(torch.sum(output_mat), input_mat)[0]
        middle_mat = middle_cube.detach().cpu().numpy()
        energy_den = exc_b3lyp + output_mat[:, 0].detach().cpu().numpy()

        vxc = (
            (0.08 + middle_mat[:, 0]) * vxc_b3lyp[0]
            + (0.19 + middle_mat[:, 1]) * vxc_b3lyp[1]
            + (0.72 + middle_mat[:, 2]) * vxc_b3lyp[2]
            + (0.81 + middle_mat[:, 3]) * vxc_b3lyp[3]
        )
        return energy_den, vxc

    def eval_xc_eff(self, rho, ni, dms, grids, coords_, mask):
        """
        Get the exc and vxc from the model, for restricted Kohn-Sham (RKS) calculations.
        Args:
            mol: PySCF molecule object.
            dms: Density matrices.
            ni: NumInt object.
            coords_: Coordinates of the grid points.
            weights_: Weights of the grid points.
        Returns:
            exc: Exchange-correlation energy.
            vxc: Exchange-correlation potential.
        """
        if self.model_type == "center_4":
            return self.eval_xc_eff_4(rho, ni, dms, grids, coords_, mask)
        elif self.model_type == "cube":
            return self.eval_xc_eff_cube(rho, ni, dms, grids, coords_, mask)
        else:
            raise ValueError(f"Unknown model {self.model_name} for eval_xc_eff")
