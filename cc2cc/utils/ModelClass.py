"""
Generate list of model.
"""

from pathlib import Path
import datetime

import numpy as np
import torch
import torch.optim as optim
from torchinfo import summary

from cc2cc.utils.env_var import MAIN_PATH, CHECKPOINTS_PATH, CUBE_SIZE
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.DataBase import DataBase
from cc2cc.utils.DataBase_c import DataBase as DataBase_c
from cc2cc.utils.checkpoint import Checkpointer


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
        self.model_name = args.model
        self.load = getattr(args, "load", "")
        self.loss_multiplier_abs = getattr(args, "loss_multiplier_abs", 1.0)
        self.loss_multiplier_atomic = getattr(args, "loss_multiplier_atomic", 1.0)

        self.iters_to_accumulate = getattr(args, "iters_to_accumulate", 1)
        self.max_norm = getattr(args, "max_norm", -1)
        self.update_counter = 0

        self.model = None
        self.model_type = None
        self.model_device = None
        self.model_dtype = None

        self.optimizer = None
        self.scheduler = None
        self.loss_ene = None
        self.loss_ene_abs = None
        self.loss_ene_atomic = None
        self.dir_checkpoint = None
        self.checkpointer = None

        self.database_train = None
        self.database_eval = None

    def init_model(self, args):
        """
        Initialize the model.
        """
        if (MAIN_PATH / f"cc2cc/utils/model/{args.model}.py").exists():
            model = getattr(
                __import__(f"cc2cc.utils.model.{args.model}", fromlist=["Model"]),
                "Model",
            )
        else:
            raise ValueError("Unknown model")

        self.model: torch.nn.Module = model().to(args.device)
        self.model_device = next(self.model.parameters()).device
        self.model_dtype = next(self.model.parameters()).dtype
        self.model_type = self.model.model_type

        if args.precision == "float64":
            self.model.double()

        self.model.fully_shard()

        if args.save_dir is not None and args.save_dir != "":
            self.dir_checkpoint = (
                CHECKPOINTS_PATH / f"checkpoint_{args.save_dir}"
            ).resolve()
            if not self.dir_checkpoint.exists():
                print(f"Directory {self.dir_checkpoint} not found. Created!")
                (self.dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)
        else:
            self.dir_checkpoint = (
                CHECKPOINTS_PATH
                / f"checkpoint_{datetime.datetime.today():%Y-%m-%d-%H-%M-%S}/"
            ).resolve()

        load_checkpoint = Path(CHECKPOINTS_PATH / f"checkpoint_{self.load}/").resolve()
        list_of_path = list(load_checkpoint.glob("*.pth"))
        load_path = load_checkpoint / f"{args.load_epoch}.pth"
        state_dict = torch.load(
            load_path,
            map_location=self.model_device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {load_path}")

        self.checkpointer = Checkpointer(self.dir_checkpoint, dcp_api=False)
        # if self.checkpointer.last_training_time is not None:
        #     self.checkpointer.load_model(self.model)

    def init_train(self, args):
        """
        Initialize the optimizer, scheduler, loss function and checkpoint_dir.
        """
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.eval_step * 50,
            eta_min=args.lr / 100,
        )

        if self.checkpointer.last_training_time is not None:
            self.checkpointer.load_optim(self.model, self.optimizer)

        if args.loss_ene == "L1Loss":
            self.loss_ene = torch.nn.L1Loss(reduction="sum")
            self.loss_ene_abs = torch.nn.L1Loss(reduction="sum")
            self.loss_ene_atomic = torch.nn.L1Loss(reduction="sum")
        elif args.loss_ene == "MSELoss":
            self.loss_ene = torch.nn.MSELoss(reduction="sum")
            self.loss_ene_abs = torch.nn.MSELoss(reduction="sum")
            self.loss_ene_atomic = torch.nn.MSELoss(reduction="sum")
        else:
            raise ValueError(f"Unknown loss function {args.loss_ene}")

    def init_database(self, args, train_str_dict, eval_str_dict):
        """
        Initialize the database.
        """
        if self.model_type == "center_4":
            input_size = (302 * 75 * 10, 4)
            database_eval = DataBase_c(eval_str_dict, args)
            database_train = DataBase_c(train_str_dict, args)
        elif self.model_type == "cube":
            input_size = (302 * 75 * 10, 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
            database_eval = DataBase(eval_str_dict, args, shuffle=False)
            database_train = DataBase(train_str_dict, args, shuffle=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        print(
            summary(
                self.model,
                input_size=input_size,
                depth=10,
                dtypes=(
                    [torch.float32] if args.precision == "float32" else [torch.float64]
                ),
                mode="train",
            )
        )

        self.database_train = database_train
        self.database_eval = database_eval

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
        target = batch["output"] * weight
        data_weight = batch["data_weight"]
        output = self.model(input_) * weight

        if if_train:
            tot_loss = self.loss_ene(
                data_weight * torch.sum(target),
                data_weight * torch.sum(output),
            )
        else:
            tot_loss = torch.tensor(0.0, device=self.model_device)
        loss_record = np.abs(torch.sum(target - output).item())

        if if_train and self.loss_multiplier_abs > 1e-8:
            tot_loss += self.loss_ene_abs(
                self.loss_multiplier_abs * data_weight * target,
                self.loss_multiplier_abs * data_weight * output,
            )
        loss_abs_record = torch.sum(torch.abs(target - output)).item()

        if if_train and self.loss_multiplier_atomic > 1e-8:
            ae_target = torch.sum(target)
            ae_output = torch.sum(output)
        loss_atomic_record = torch.sum(target - output)

        for i_system in range(len(batch["atomic_systems"])):
            system_atom = batch["atomic_systems"][i_system]
            if system_atom in self.database_train.atomic_name_dict:
                name_atom = self.database_train.atomic_name_dict[system_atom]
            else:
                print(
                    f"Warning: {system_atom} not found in atomic_name_dict, "
                    "skipping atomic energy calculation."
                )
                if if_train and self.loss_multiplier_atomic > 1e-8:
                    ae_target = torch.zeros_like(ae_target)
                    ae_output = torch.zeros_like(ae_output)
                break

            atomic_batch = self.database_train.data_gpu.dataset.get_from_name(name_atom)
            atomic_batch = self.database_train.process_batch_dataset(atomic_batch)

            atomic_input_ = atomic_batch["input"]
            atomic_weight = atomic_batch["weight"]
            atomic_target = atomic_batch["output"] * atomic_weight
            atomic_output = self.model(atomic_input_) * atomic_weight

            if if_train and self.loss_multiplier_atomic > 1e-8:
                ae_target -= (
                    torch.sum(atomic_target) * batch["atomic_stoichiometry"][i_system]
                )
                ae_output -= (
                    torch.sum(atomic_output) * batch["atomic_stoichiometry"][i_system]
                )
            loss_atomic_record -= (
                torch.sum(atomic_target - atomic_output)
                * batch["atomic_stoichiometry"][i_system]
            )

        if if_train and self.loss_multiplier_atomic > 1e-8:
            tot_loss += self.loss_ene_atomic(
                self.loss_multiplier_atomic * data_weight * ae_target,
                self.loss_multiplier_atomic * data_weight * ae_output,
            )
        loss_atomic_record = torch.abs(loss_atomic_record).item()

        tot_loss = tot_loss / self.iters_to_accumulate
        data_record = {
            "loss_ene": AU2KCALMOL * loss_record,
            "loss_ene_abs": AU2KCALMOL * loss_abs_record,
            "loss_ene_atomic": AU2KCALMOL * loss_atomic_record,
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
            batch = self.database_train.process_batch(batch)

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
            batch = self.database_train.process_batch(batch)

            with torch.no_grad():
                data_record = self.loss(batch, if_train=False)

            data_record_l.append(data_record)

        return data_record_l

    def save_model(self, epoch):
        """
        Save the model to the checkpoint.
        """
        state_dict = self.model.state_dict()
        torch.save(state_dict, self.dir_checkpoint / f"{epoch}.pth")

    def eval_xc_eff_cube(
        self,
        mol,
        dms,
        rho,
        ni,
        grids,
        weights_,
        coords_,
    ):
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
        if mol.spin == 0:
            exc_b3lyp, rho_cube, vxc_b3lyp = grids.gen_cube_rho_rks(
                mol, dms, rho, ni=ni, coords=coords_, weights=weights_, require_vxc=True
            )
        else:
            exc_b3lyp, rho_cube, vxc_b3lyp = grids.gen_cube_rho_uks(
                mol, dms, rho, ni=ni, coords=coords_, weights=weights_, require_vxc=True
            )

        input_mat = torch.tensor(
            rho_cube,
            dtype=self.model_dtype,
            device=self.model_device,
        )
        input_mat.requires_grad = True
        output_mat = self.model(input_mat)[:, 0]

        middle_cube = torch.autograd.grad(
            torch.sum(output_mat),
            input_mat,
            create_graph=True,
        )[0]

        middle_mat = grids.get_center_density(middle_cube).detach().cpu().numpy()
        energy_den = exc_b3lyp + output_mat.detach().cpu().numpy()

        vxc = (
            (0.08 + middle_mat[:, 0]) * vxc_b3lyp[0]
            + (0.19 + middle_mat[:, 1]) * vxc_b3lyp[1]
            + (0.72 + middle_mat[:, 2]) * vxc_b3lyp[2]
            + (0.81 + middle_mat[:, 3]) * vxc_b3lyp[3]
        )
        return energy_den, vxc

    def eval_xc_eff_4(
        self,
        mol,
        dms,
        rho,
        ni,
        grids,
        weights_,
        coords_,
    ):
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
        if mol.spin == 0:
            exc_b3lyp, rho_b3lyp, vxc_b3lyp = grids.gen_rho_rks(
                mol, dms, rho, ni=ni, coords=coords_, weights=weights_, require_vxc=True
            )
        else:
            exc_b3lyp, rho_b3lyp, vxc_b3lyp = grids.gen_rho_uks(
                mol, dms, rho, ni=ni, coords=coords_, weights=weights_, require_vxc=True
            )

        input_mat = torch.tensor(
            rho_b3lyp,
            dtype=self.model_dtype,
            device=self.model_device,
        )
        input_mat.requires_grad = True
        output_mat = self.model(input_mat)[:, 0]

        middle_cube = torch.autograd.grad(
            torch.sum(output_mat),
            input_mat,
            create_graph=True,
        )[0]

        middle_mat = middle_cube.detach().cpu().numpy()
        energy_den = exc_b3lyp + output_mat.detach().cpu().numpy()

        vxc = (
            (0.08 + middle_mat[:, 0]) * vxc_b3lyp[0]
            + (0.19 + middle_mat[:, 1]) * vxc_b3lyp[1]
            + (0.72 + middle_mat[:, 2]) * vxc_b3lyp[2]
            + (0.81 + middle_mat[:, 3]) * vxc_b3lyp[3]
        )
        return energy_den, vxc

    def eval_xc_eff(
        self,
        mol,
        dms,
        rho,
        ni,
        grids,
        weights_,
        coords_,
    ):
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
            return self.eval_xc_eff_4(mol, dms, rho, ni, grids, weights_, coords_)
        elif self.model_type == "cube":
            return self.eval_xc_eff_cube(mol, dms, rho, ni, grids, weights_, coords_)
        else:
            raise ValueError(f"Unknown model {self.model_name} for eval_xc_eff")
