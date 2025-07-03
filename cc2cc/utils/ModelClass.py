"""
Generate list of model.
"""

from pathlib import Path
import datetime
import os
import numpy as np
import torch
import torch.optim as optim
from torchinfo import summary

from cc2cc.utils.env_var import MAIN_PATH, CHECKPOINTS_PATH, CUBE_SIZE, DEEPSPEED
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.DataBaseCube import DataBaseCube
from cc2cc.utils.DataBaseCenter import DataBaseCenter

if DEEPSPEED:
    import deepspeed


class ModelClass:
    """
    Model_Class
    """

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    loss_ene: torch.nn.Module
    loss_ene_abs: torch.nn.Module
    loss_ene_atomic: torch.nn.Module
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
        if self.args.dist_url == "env://" and self.args.world_size == -1:
            self.args.world_size = int(os.environ["WORLD_SIZE"])
        self.distributed = (
            self.args.world_size > 1
            or self.args.multiprocessing_distributed
            or self.args.deepspeed
        )
        if self.args.deepspeed:
            self.args.gpu = self.args.local_rank
        else:
            self.args.gpu = self.args.device
        self.load = self.args.load
        self.loss_multiplier_abs = self.args.loss_multiplier_abs
        self.loss_multiplier_atomic = self.args.loss_multiplier_atomic

        self.iters_to_accumulate = self.args.iters_to_accumulate
        self.max_norm = self.args.max_norm
        self.update_counter = 0

        self.dir_checkpoint = None
        self.checkpointer = None

    def init_model(self):
        """
        Initialize the model.
        """
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

        if self.args.deepspeed:
            deepspeed.init_distributed()

        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype
        self.model_type = self.model.model_type

        if self.args.save_dir is not None and self.args.save_dir != "":
            self.dir_checkpoint = (
                CHECKPOINTS_PATH / f"checkpoint_{self.args.save_dir}"
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
        load_path = load_checkpoint / f"{self.args.load_epoch}.pth"
        if load_path.exists():
            print(f"Loading model from {load_path}")
            state_dict = torch.load(
                load_path,
                map_location=self.device,
                weights_only=True,
            )
            if "module" in list(state_dict.keys())[0]:
                # For deepspeed or distributed training
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            self.model.load_state_dict(state_dict)
        else:
            print(f"Model {load_path} not found, starting from scratch.")

    def init_train(self):
        """
        Initialize the optimizer, scheduler, loss function and checkpoint_dir.
        """
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.eval_step * 50,
            eta_min=self.args.lr / 100,
        )

        if self.args.loss_ene == "L1Loss":
            self.loss_ene = torch.nn.L1Loss(reduction="sum").cuda(self.args.gpu)
            self.loss_ene_abs = torch.nn.L1Loss(reduction="sum").cuda(self.args.gpu)
            self.loss_ene_atomic = torch.nn.L1Loss(reduction="sum").cuda(self.args.gpu)
        elif self.args.loss_ene == "MSELoss":
            self.loss_ene = torch.nn.MSELoss(reduction="sum").cuda(self.args.gpu)
            self.loss_ene_abs = torch.nn.MSELoss(reduction="sum").cuda(self.args.gpu)
            self.loss_ene_atomic = torch.nn.MSELoss(reduction="sum").cuda(self.args.gpu)
        else:
            raise ValueError(f"Unknown loss function {self.args.loss_ene}")

        if self.args.deepspeed:
            self.model, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                optimizer=self.optimizer,
                args=self.args,
                dist_init_required=False,
            )

    def init_database(self, train_str_dict, eval_str_dict):
        """
        Initialize the database.
        """
        if self.model_type == "center_4":
            input_size = (302 * 75 * 10, 4)
            self.database_eval = DataBaseCenter(
                eval_str_dict, self.args, shuffle=False, distributed=self.distributed
            )
            self.database_train = DataBaseCenter(
                train_str_dict, self.args, distributed=self.distributed
            )
        elif self.model_type == "cube":
            input_size = (302 * 75 * 10, 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
            self.database_eval = DataBaseCube(
                eval_str_dict, self.args, shuffle=False, distributed=self.distributed
            )
            self.database_train = DataBaseCube(
                train_str_dict, self.args, distributed=self.distributed
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        print(
            summary(
                self.model,
                input_size=input_size,
                depth=10,
                dtypes=(
                    [torch.float32]
                    if self.args.precision == "float32"
                    else [torch.float64]
                ),
                mode="train",
            )
        )

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

    def loss(self, batch, if_train=True, device=None):
        """
        Calculate the loss.
        ae if for atomic energy.
        """
        input_ = batch["input"]
        weight = batch["weight"]
        target = batch["output"] * weight
        data_weight = batch["data_weight"]
        output = self.model(input_) * weight

        tot_loss = self.loss_ene(
            data_weight * torch.sum(target),
            data_weight * torch.sum(output),
        )
        loss_record = np.abs(torch.sum(target - output).item())

        if self.loss_multiplier_abs > 1e-8:
            tot_loss += self.loss_ene_abs(
                self.loss_multiplier_abs * data_weight * target,
                self.loss_multiplier_abs * data_weight * output,
            )
        loss_abs_record = torch.sum(torch.abs(target - output)).item()

        if self.loss_multiplier_atomic > 1e-8:
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
                if self.loss_multiplier_atomic > 1e-8:
                    ae_target = torch.zeros_like(ae_target)
                    ae_output = torch.zeros_like(ae_output)
                break

            atomic_batch = self.database_train.dataset.get_from_name(name_atom)
            atomic_batch = self.database_train.process_batch_dataset(
                atomic_batch, device=device
            )

            atomic_input_ = atomic_batch["input"]
            atomic_weight = atomic_batch["weight"]
            atomic_target = atomic_batch["output"] * atomic_weight
            atomic_output = self.model(atomic_input_) * atomic_weight

            if self.loss_multiplier_atomic > 1e-8:
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

        if self.loss_multiplier_atomic > 1e-8:
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
            batch = self.database_train.process_batch(batch, device=self.args.gpu)
            tot_loss, data_record = self.loss(batch, device=self.args.gpu)

            if self.args.deepspeed:
                self.model.backward(tot_loss)
                self.model.step()
            else:
                tot_loss.backward()
                self.update_counter += 1
                if self.update_counter % self.iters_to_accumulate == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_norm
                    )
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
            batch = self.database_train.process_batch(batch, device=self.args.gpu)

            with torch.no_grad():
                data_record = self.loss(batch, if_train=False, device=self.args.gpu)

            data_record_l.append(data_record)

        return data_record_l

    def save_model(self, epoch):
        """
        Save the model to the checkpoint.
        """
        if (
            not self.args.multiprocessing_distributed
            or (
                self.args.multiprocessing_distributed
                and self.args.rank % torch.cuda.device_count() == 0
            )
            or self.args.local_rank == 0
        ):
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
            dtype=self.dtype,
            device=self.device,
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
            dtype=self.dtype,
            device=self.device,
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
