"""
Generate list of model.
"""

from pathlib import Path
import datetime

import numpy as np
import torch
import torch.optim as optim

import pyscf

from cc2cc.utils.env_var import CHECKPOINTS_PATH, CUBE_MIDDLE
from cc2cc.utils.mol import AU2KCALMOL, AU2DEBYE
from cc2cc.utils.Grids import Grid

from cc2cc.utils.model.cnn3d import Model


class ModelDict:
    """
    Model_Dict
    """

    def __init__(
        self,
        load,
        device,
        precision,
        with_eval=True,
        if_mkdir=True,
        load_epoch=-1,
    ):
        """
        input:
        output:
            model_dict: dictionary of models
        """
        self.load = load
        self.with_eval = with_eval
        self.load_epoch = load_epoch

        self.device = device
        if precision == "float32":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float64

        self.dir_checkpoint = Path(
            CHECKPOINTS_PATH
            / f"checkpoint-ccdft_{datetime.datetime.today():%Y-%m-%d-%H-%M-%S}/"
        ).resolve()
        if if_mkdir:
            print(f"Create checkpoint directory: {self.dir_checkpoint}")
            self.dir_checkpoint.mkdir(parents=True, exist_ok=True)
            (self.dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)

        self.model_dict = {}
        self.model_dict["size"] = {}
        self.optimizer_dict = {}
        self.scheduler_dict = {}

        self.model: torch.nn.Module = Model().to(device)
        if precision == "float64":
            self.model.double()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4)

        if self.with_eval:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=10,
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=1000,
                eta_min=1e-6,
            )

        self.loss_multiplier = 0.1
        self.loss_ene = torch.nn.L1Loss(reduction="none")
        self.loss_ene_tot = torch.nn.L1Loss(reduction="sum")

    def load_model(self):
        """
        Load the model from the checkpoint.
        """
        if self.load not in ["", "None", "NEW", "new"]:
            load_checkpoint = Path(
                CHECKPOINTS_PATH / f"checkpoint-ccdft_{self.load}/"
            ).resolve()
            if load_checkpoint.exists():
                print(f"Loading from {load_checkpoint}")
                list_of_path = list(load_checkpoint.glob("*.pth"))
                if len(list_of_path) == 0:
                    print("No model found, use random initialization.")
                load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
                if self.load_epoch != -1:
                    load_path = load_checkpoint / f"{self.load_epoch}.pth"
                state_dict = torch.load(load_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Model loaded from {load_path}")
            else:
                print(f"Load checkpoint directory {load_checkpoint} not found.")

    def train(self):
        """
        Set the model to train mode.
        """
        self.model.train(True)
        self.optimizer.zero_grad(set_to_none=True)

    def zero_grad(self):
        """
        Set the model to train mode.
        """
        self.optimizer.zero_grad(set_to_none=True)

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        """
        Step the optimizer.
        """
        self.optimizer.step()

    def loss(self, batch):
        """
        Calculate the loss.
        """
        input_mat = batch["input"]
        weight = batch["weight"]
        output_mat_real = batch["output"]
        output_mat = self.model(input_mat)
        loss_ene_mat = self.loss_ene(output_mat, output_mat_real)
        # loss_ene_mat = self.loss_ene(
        #     output_mat_real[:, 0]
        #     * (
        #         input_mat[:, 0, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        #         + input_mat[:, 1, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        #     )
        #     * weight[:, 0],
        #     output_mat[:, 0]
        #     * (
        #         input_mat[:, 0, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        #         + input_mat[:, 1, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        #     )
        #     * weight[:, 0],
        # )
        loss_ene = torch.mean(loss_ene_mat)

        loss_ene_tot = self.loss_ene_tot(
            torch.sum(
                output_mat_real[:, 0]
                * (
                    input_mat[:, 0, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
                    + input_mat[:, 1, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
                )
                * weight[:, 0]
            ),
            torch.sum(
                output_mat[:, 0]
                * (
                    input_mat[:, 0, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
                    + input_mat[:, 1, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
                )
                * weight[:, 0]
            ),
        )

        return loss_ene, loss_ene_tot

    def tot_loss(self, loss_ene, loss_ene_tot):
        """
        Calculate the total loss.
        """
        return loss_ene + self.loss_multiplier * loss_ene_tot

    def save_model(self, epoch):
        """
        Save the model to the checkpoint.
        """
        state_dict = self.model.state_dict()
        torch.save(state_dict, self.dir_checkpoint / f"{epoch}.pth")

    def train_model(self, database_train):
        """
        Train the model, one epoch.
        """
        self.train()
        loss_ene_l, loss_ene_tot_l = [], []
        database_train.rng.shuffle(database_train.name_list)

        for name in database_train.name_list:
            loss_ene_name = 0.0
            number_batch_name = 0
            loss_ene_tot_name = 0.0

            for batch in database_train.data_gpu[name]:
                self.zero_grad()
                loss_ene, loss_ene_tot = self.loss(batch)
                self.tot_loss(loss_ene, loss_ene_tot).backward()
                self.step()
                number_batch_name += len(batch["weight"])
                loss_ene_name += loss_ene.item() * len(batch["weight"])
                loss_ene_tot_name += loss_ene_tot.item()

            loss_ene_l.append(AU2KCALMOL * loss_ene_name / number_batch_name)
            loss_ene_tot_l.append(AU2KCALMOL * np.abs(loss_ene_tot_name))

        return np.array(loss_ene_l), np.array(loss_ene_tot_l)

    def eval_model(self, database_eval):
        """
        Evaluate the model.
        """
        self.eval()
        loss_ene_l, loss_ene_tot_l = [], []

        for name in database_eval.name_list:
            loss_ene_name = 0.0
            number_batch_name = 0
            loss_ene_tot_name = 0.0

            for batch in database_eval.data_gpu[name]:
                with torch.no_grad():
                    loss_ene, loss_ene_tot = self.loss(batch)
                number_batch_name += len(batch["weight"])
                loss_ene_name += loss_ene.item() * len(batch["weight"])
                loss_ene_tot_name += loss_ene_tot.item()

            loss_ene_l.append(AU2KCALMOL * loss_ene_name / number_batch_name)
            loss_ene_tot_l.append(AU2KCALMOL * np.abs(loss_ene_tot_name))

        return np.array(loss_ene_l), np.array(loss_ene_tot_l)

    def get_e(
        self,
        dft: pyscf.dft.rks.RKS,
        grids: Grid,
        dms: np.ndarray = None,
    ):
        """
        Obtain the energy density.
        Input: dft instance and grids instance.
        Output: the potential (ngrids).
        """
        if dms is None:
            dms = dft.make_rdm1()

        return

    def get_e_density(
        self,
        dft: pyscf.dft.rks.RKS,
        grids: Grid,
        dms: np.ndarray = None,
    ):
        """
        Obtain the energy density.
        Input: dft instance and grids instance.
        Output: the potential (ngrids).
        """
        if dms is None:
            dms = dft.make_rdm1()

        return

    def get_v(
        self,
        dft: pyscf.dft.rks.RKS,
        grids: Grid,
        dms: np.ndarray = None,
    ):
        """
        Obtain the energy density.
        Input: dft instance and grids instance.
        Output: the potential (ngrids).
        """
        if dms is None:
            dms = dft.make_rdm1()

        return
