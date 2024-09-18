"""
Generate list of model.
"""

from pathlib import Path
import datetime
import importlib.resources
import os

import numpy as np
import torch
import torch.optim as optim

from dft2cc.utils.model.cnn_fc import CNN_FC as Model
from dft2cc.utils.env_var import CHECKPOINTS_PATH
from dft2cc.utils.DataBase import process_input


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

        self.model = Model().to(device)

        if precision == "float64":
            self.model.double()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        if self.with_eval:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=10,
            )
        else:
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=1.0 - 1e-6,
            )

        self.loss_multiplier = 1.0

        self.loss_fn1 = torch.nn.L1Loss()
        self.loss_fn2 = torch.nn.L1Loss()
        self.loss_fn3 = torch.nn.L1Loss(reduction="sum")

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
                list_of_path = list(load_checkpoint.glob(f"*.pth"))
                if len(list_of_path) == 0:
                    print("No model found, use random initialization.")
                load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
                if self.load_epoch != -1:
                    load_path = load_checkpoint / f"{self.load_epoch}.pth"
                state_dict = torch.load(
                    load_path, map_location=self.device, weights_only=True
                )
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
        output_mat_real = batch["output"]
        error_energy = batch["error_energy"]
        error_dipole = batch["error_dipole"]

        output_mat = self.model(input_mat)

        loss_ene = self.loss_fn1(error_energy, torch.sum(output_mat[:, 0]))
        loss_dipole = self.loss_fn1(error_dipole, torch.sum(output_mat[:, 1:4]))
        loss_force = self.loss_fn1(output_mat_real, output_mat[:, 3:6])

        return loss_ene, loss_dipole, loss_force

    def tot_loss(self, train_loss_ene, train_loss_dipole, train_loss_force):
        """
        Calculate the total loss.
        """
        return 10 * train_loss_ene + train_loss_dipole + train_loss_force

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

        loss_ene_l, loss_dipole_l, loss_force_l = [], [], []
        database_train.rng.shuffle(database_train.name_list)

        for name in database_train.name_list:
            self.zero_grad()

            # only one batch
            for batch in database_train.data_gpu[name]:
                loss_ene, loss_dipole, loss_force = self.loss(batch)

            loss_ene_l.append(loss_ene.item())
            loss_dipole_l.append(loss_dipole.item())
            loss_force_l.append(loss_force.item())

            loss_tot = self.tot_loss(loss_ene, loss_dipole, loss_force)
            loss_tot.backward()

            self.step()

        return np.array(loss_ene_l), np.array(loss_dipole_l), np.array(loss_force_l)

    def eval_model(self, database_eval):
        """
        Evaluate the model.
        """
        self.eval()

        loss_ene_l, loss_dipole_l, loss_force_l = [], [], []

        for name in database_eval.name_list:
            self.zero_grad()

            # only one batch
            for batch in database_eval.data_gpu[name]:
                with torch.no_grad():
                    loss_ene, loss_dipole, loss_force = self.loss(batch)

            loss_ene_l.append(loss_ene.item())
            loss_dipole_l.append(loss_dipole.item())
            loss_force_l.append(loss_force.item())

        return np.array(loss_ene_l), np.array(loss_dipole_l), np.array(loss_force_l)

    def get_val(self, scf_r_3, grids):
        """
        Obtain the potential.
        Input: [rho, nabla rho] (4, ngrids),
        Output: the potential (ngrids).
        """
        if len(np.shape(scf_r_3)) == 2:
            input_mat = process_input(scf_r_3, grids)
        elif len(np.shape(scf_r_3)) == 1:
            input_mat = grids.vector_to_matrix(scf_r_3)
        else:
            raise ValueError("scf_r_3 must be lda or gga density")
        input_mat = torch.tensor(input_mat[:, np.newaxis, :, :], dtype=self.dtype).to(
            "cuda"
        )

        with torch.no_grad():
            output_mat = self.model(input_mat).detach().cpu().numpy()

        correct_ene = torch.sum(output_mat[:, 6])
        correct_dipole = output_mat[:, 0:3]
        loss_force = output_mat[:, 3:6]

        return correct_ene, correct_dipole, loss_force
