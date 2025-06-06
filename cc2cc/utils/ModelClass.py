"""
Generate list of model.
"""

from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from cc2cc.utils.env_var import MAIN_PATH, CHECKPOINTS_PATH
from cc2cc.utils.mol import AU2KCALMOL


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
        self.with_eval = getattr(args, "with_eval", True)
        self.loss_multiplier_abs = getattr(args, "loss_multiplier_abs", 1.0)
        self.loss_multiplier_atomic = getattr(args, "loss_multiplier_atomic", 1.0)

        self.iters_to_accumulate = getattr(args, "iters_to_accumulate", 1)
        self.max_norm = getattr(args, "max_norm", -1)
        self.update_counter = 0

        self.model = None
        self.model_type = None

        self.optimizer = None
        self.scheduler = None
        self.loss_ene = None
        self.loss_ene_abs = None
        self.dir_checkpoint = None

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
        self.model_type = self.model.model_type

        if args.precision == "float64":
            self.model.double()

    def init_train(self, args):
        """
        Initialize the optimizer, scheduler, loss function and checkpoint_dir.
        """
        if self.with_eval:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=50,
            )
        else:
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

        if args.loss_ene == "L1Loss":
            self.loss_ene = torch.nn.L1Loss(reduction="sum")
        elif args.loss_ene == "MSELoss":
            self.loss_ene = torch.nn.MSELoss(reduction="sum")
        else:
            raise ValueError(f"Unknown loss function {args.loss_ene}")

        if args.loss_ene_abs == "L1Loss":
            self.loss_ene_abs = torch.nn.L1Loss(reduction="sum")
        elif args.loss_ene_abs == "MSELoss":
            self.loss_ene_abs = torch.nn.MSELoss(reduction="sum")
        else:
            raise ValueError(f"Unknown loss function {args.loss_ene_abs}")

        if args.save_dir is not None and args.save_dir != "":
            self.dir_checkpoint = (
                CHECKPOINTS_PATH / f"checkpoint-ccdft_{args.basis}_{args.save_dir}"
            ).resolve()
            if not self.dir_checkpoint.exists():
                print(f"Directory {self.dir_checkpoint} not found. Created!")
                (self.dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)
        else:
            self.dir_checkpoint = (
                CHECKPOINTS_PATH
                / f"checkpoint-ccdft_{args.basis}_{datetime.datetime.today():%Y-%m-%d-%H-%M-%S}/"
            ).resolve()

    def init_database(self, database_train, database_eval):
        """
        Initialize the database.
        """
        self.database_train = database_train
        self.database_eval = database_eval

    def load_model(self, args):
        """
        Load the model from the checkpoint.
        """
        load_checkpoint = Path(
            CHECKPOINTS_PATH / f"checkpoint-ccdft_{args.basis}_{self.load}/"
        ).resolve()

        list_of_path = list(load_checkpoint.glob("*.pth"))

        if len(list_of_path) == 0:
            print(f"No model found in {load_checkpoint}, use random initialization.")
        else:
            if args.load_epoch == -1:
                load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
            else:
                load_path = load_checkpoint / f"{args.load_epoch}.pth"
            state_dict = torch.load(
                load_path,
                map_location=next(self.model.parameters()).device,
                weights_only=True,
            )
            self.model.load_state_dict(state_dict)
            print(f"Model loaded from {load_path}")

    def train(self):
        """
        Set the model to train mode.
        """
        self.model.train(True)
        self.optimizer.zero_grad(set_to_none=True)

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)

    def update(self):
        """
        Update the model.

        # See https://kozodoi.me/blog/20210219/gradient-accumulation and
        # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
        """
        self.update_counter += 1

        if self.update_counter % self.iters_to_accumulate != 0:
            return

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.update_counter = 0

    def tot_loss(self, loss_ene, loss_ene_abs, loss_ene_atomic=None):
        """
        Calculate the total loss.
        """
        if isinstance(self.loss_ene, torch.nn.L1Loss):
            if loss_ene_atomic is not None:
                tot_loss = (
                    loss_ene
                    + loss_ene_abs * self.loss_multiplier_abs
                    + loss_ene_atomic * self.loss_multiplier_atomic
                )
            else:
                tot_loss = loss_ene + loss_ene_abs * self.loss_multiplier_abs
        elif isinstance(self.loss_ene, torch.nn.MSELoss):
            if loss_ene_atomic is not None:
                tot_loss = (
                    loss_ene
                    + loss_ene_abs * self.loss_multiplier_abs**2
                    + loss_ene_atomic * self.loss_multiplier_atomic**2
                )
            else:
                tot_loss = loss_ene + loss_ene_abs * self.loss_multiplier_abs**2
        else:
            raise ValueError("Unknown loss function")

        return tot_loss

    def loss(self, batch, data_weight=1.0):
        """
        Calculate the loss.
        """
        input_mat = batch["input"]
        weight = batch["weight"]
        output_mat_real = batch["output"] * weight
        data_weight = 1 / np.sqrt(data_weight)
        # data_weight = 1
        # data_weight = np.sqrt(data_weight)

        output_mat = self.model(input_mat) * weight

        loss_ene = self.loss_ene(
            data_weight * torch.sum(output_mat_real),
            data_weight * torch.sum(output_mat),
        )
        loss_ene_abs = self.loss_ene_abs(
            data_weight * output_mat_real,
            data_weight * output_mat,
        )

        loss_record = np.abs(torch.sum(output_mat_real - output_mat).item())
        loss_abs_record = np.abs(
            torch.sum(torch.abs(output_mat_real - output_mat)).item()
        )
        loss_atomic_record = torch.sum(output_mat_real - output_mat)

        if self.database_train is not None:
            atomic_energy_pred = torch.sum(output_mat)
            atomic_energy_real = torch.sum(output_mat_real)
            for i_system in range(len(batch["atomic_systems"])):
                system_atom = batch["atomic_systems"][i_system]
                if system_atom in self.database_train.atomic_name_dict:
                    name_atom = self.database_train.atomic_name_dict[system_atom]
                else:
                    print(
                        f"Warning: {system_atom} not found in atomic_name_dict, "
                        "skipping atomic energy calculation."
                    )
                    break
                atomic_batch = self.database_train.data_gpu[name_atom]

                if not self.database_train.if_load_to_gpu_once:
                    atomic_batch = self.database_train.process_batch(atomic_batch)

                atomic_input_mat = atomic_batch["input"]
                atomic_weight = atomic_batch["weight"]
                atomic_output_mat_real = atomic_batch["output"] * atomic_weight

                atomic_output_mat = self.model(atomic_input_mat) * atomic_weight

                atomic_energy_pred -= (
                    torch.sum(atomic_output_mat)
                    * batch["atomic_stoichiometry"][i_system]
                )
                atomic_energy_real -= (
                    torch.sum(atomic_output_mat_real)
                    * batch["atomic_stoichiometry"][i_system]
                )
                loss_atomic_record -= (
                    torch.sum(atomic_output_mat_real - atomic_output_mat)
                    * batch["atomic_stoichiometry"][i_system]
                )
            else:
                # If we break the loop, we set the atomic energy loss to zero
                atomic_energy_pred = torch.tensor(
                    0.0, device=next(self.model.parameters()).device
                )
                atomic_energy_real = torch.tensor(
                    0.0, device=next(self.model.parameters()).device
                )

            loss_ene_atomic = self.loss_ene(
                data_weight * atomic_energy_real,
                data_weight * atomic_energy_pred,
            )
            loss_atomic_record = torch.abs(loss_atomic_record).item()

            tot_loss = (
                self.tot_loss(loss_ene, loss_ene_abs, loss_ene_atomic)
                / self.iters_to_accumulate
            )
        else:
            tot_loss = self.tot_loss(loss_ene, loss_ene_abs) / self.iters_to_accumulate

        data_record = {
            "loss_ene": AU2KCALMOL * loss_record,
            "loss_ene_abs": AU2KCALMOL * loss_abs_record,
            "loss_ene_atomic": AU2KCALMOL * loss_atomic_record,
            "loss_tot": AU2KCALMOL * tot_loss.item(),
        }

        return tot_loss, data_record

    def save_model(self, epoch):
        """
        Save the model to the checkpoint.
        """
        state_dict = self.model.state_dict()
        torch.save(state_dict, self.dir_checkpoint / f"{epoch}.pth")

    def train_model(self):
        """
        Train the model, one epoch.
        1 / self.iters_to_accumulate is the effective batch size.
        See https://kozodoi.me/blog/20210219/gradient-accumulation and
        https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
        """
        self.train()
        data_record_l = []
        self.database_train.shuffle()

        for name in self.database_train.name_list:
            batch = self.database_train.data_gpu[name]
            data_weight = self.database_train.data_weight[name]

            if not self.database_train.if_load_to_gpu_once:
                batch = self.database_train.process_batch(batch)

            tot_loss, data_record = self.loss(batch, data_weight)
            tot_loss.backward()
            self.update()

            data_record["name"] = name
            data_record_l.append(data_record)

        return data_record_l

    def eval_model(self):
        """
        Evaluate the model.
        """
        self.eval()
        data_record_l = []

        for name in self.database_eval.name_list:
            batch = self.database_eval.data_gpu[name]
            data_weight = self.database_eval.data_weight[name]

            if not self.database_eval.if_load_to_gpu_once:
                batch = self.database_eval.process_batch(batch)

            with torch.no_grad():
                _, data_record = self.loss(batch, data_weight)

            data_record["name"] = name
            data_record_l.append(data_record)

        return data_record_l

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
            dtype=next(self.model.parameters()).dtype,
            device=next(self.model.parameters()).device,
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
            dtype=next(self.model.parameters()).dtype,
            device=next(self.model.parameters()).device,
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
