"""
Generate list of model.
"""

from pathlib import Path
import datetime

import numpy as np
import torch
import torch.optim as optim

import pyscf
from pyscf.dft.numint import _rks_gga_wv0, _scale_ao, _dot_ao_ao
from pyscf.dft.libxc import xc_type

from cc2cc.utils.env_var import CHECKPOINTS_PATH
from cc2cc.utils.mol import AU2KCALMOL, AU2DEBYE

from cc2cc.utils.Grids import Grid
from cc2cc.utils.model.cnn3d import Model


class ModelDict:
    """
    Model_Dict
    """

    def __init__(self, args):
        """
        input:
        output:
            model_dict: dictionary of models
        """
        self.load = args.load
        self.with_eval = args.with_eval
        self.load_epoch = args.load_epoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        self.dir_checkpoint = Path(
            CHECKPOINTS_PATH
            / f"checkpoint-ccdft_{datetime.datetime.today():%Y-%m-%d-%H-%M-%S}/"
        ).resolve()

        self.model: torch.nn.Module = Model().to(self.device)
        if args.precision == "float64":
            self.dtype = torch.float64
            self.model.double()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

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

        self.loss_multiplier = 1
        self.loss_ene = torch.nn.L1Loss(reduction="none")
        self.loss_ene_tot = torch.nn.L1Loss(reduction="sum")

    def load_model(self):
        """
        Load the model from the checkpoint.
        """
        load_checkpoint = Path(
            CHECKPOINTS_PATH / f"checkpoint-ccdft_{self.load}/"
        ).resolve()
        list_of_path = list(load_checkpoint.glob("*.pth"))
        if len(list_of_path) == 0:
            print("No model found, use random initialization.")
            if not load_checkpoint.exists():
                print(f"Directory {load_checkpoint} not found. Created!")
                (self.dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)
        else:
            if self.load_epoch == -1:
                load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
            else:
                load_path = load_checkpoint / f"{self.load_epoch}.pth"
            state_dict = torch.load(
                load_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)
            print(f"Model loaded from {load_path}")

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
        loss_ene_mat = self.loss_ene(
            output_mat * weight,
            output_mat_real * weight,
        )
        loss_ene = torch.sum(loss_ene_mat)

        loss_ene_tot = self.loss_ene_tot(
            torch.sum(output_mat_real * weight),
            torch.sum(output_mat * weight),
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
        ks: pyscf.dft.rks.RKS,
        grids: Grid,
        dms: np.ndarray = None,
    ):
        """
        Obtain the energy density.
        Input: dft instance and grids instance.
        Output: the potential (ngrids).
        """
        if dms is None:
            dms = ks.make_rdm1()

        rho_cube = grids.gen_cube_rho(ks.mol, dms)
        input_mat = torch.tensor(rho_cube, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            output_mat = self.model(input_mat)
        exc_cc_grids = output_mat.detach().cpu().numpy()[:, 0]

        return np.sum(exc_cc_grids * grids.weights)

    def get_nev(
        self,
        ni,
        ks: pyscf.dft.rks.RKS,
        grids: Grid,
        dms,
        xc_code="b3lyp",
        hermi=1,
        max_memory=2000,
    ):
        """
        Obtain the energy density.
        Input: dft instance and grids instance.
        Output: the potential (ngrids).
        """
        rho_cube = grids.gen_cube_rho(ks.mol, dms, reset=True)
        input_mat = torch.tensor(rho_cube, dtype=self.dtype, device=self.device)
        input_mat.requires_grad = True
        output_mat = self.model(input_mat)[:, 0]

        middle_cube = torch.autograd.grad(
            torch.sum(output_mat),
            input_mat,
            create_graph=True,
        )[0]
        middle_mat = grids.get_center_density(middle_cube).detach().cpu().numpy()
        excsum = (output_mat.detach().cpu().numpy() * grids.weights).sum()

        if ks.mol.spin == 0:
            vrho = (middle_mat[:, 0] + middle_mat[:, 1]) / 2
        else:
            vrho = np.array((middle_mat[:, 0], middle_mat[:, 1]))

        ao = ni.eval_ao(ks.mol, grids.coords, deriv=1)
        rho = ni.eval_rho(ks.mol, ao, dms, xctype=xc_type(xc_code))
        if ks.mol.spin == 0:
            lda_grids = pyscf.dft.libxc.eval_xc("LDA,", rho[0], 0)
            vwn_grids = pyscf.dft.libxc.eval_xc(",VWN3", rho[0], 0)
            b88_grids = pyscf.dft.libxc.eval_xc("B88,", rho, 0)
            lyp_grids = pyscf.dft.libxc.eval_xc(",LYP", rho, 0)
        else:
            lda_grids = pyscf.dft.libxc.eval_xc("LDA,", rho[0], 1)
            vwn_grids = pyscf.dft.libxc.eval_xc(",VWN3", rho[0], 1)
            b88_grids = pyscf.dft.libxc.eval_xc("B88,", rho, 1)
            lyp_grids = pyscf.dft.libxc.eval_xc(",LYP", rho, 1)

        # hyb_coeff = [0.08, 0.19, 0.72, 0.81]
        hyb_coeff = [0.0, 0.0, 0.0, 0.0]
        exc_b3lyp = (
            hyb_coeff[0] * lda_grids[0]
            + hyb_coeff[1] * vwn_grids[0]
            + hyb_coeff[2] * b88_grids[0]
            + hyb_coeff[3] * lyp_grids[0]
        )
        vrho = hyb_coeff[0] * lda_grids[1][0] + hyb_coeff[1] * vwn_grids[1][0]
        vrho += (hyb_coeff[2] + middle_mat[:, 2]) * b88_grids[1][0]
        vrho += (hyb_coeff[3] + middle_mat[:, 3]) * lyp_grids[1][0]
        vsigma = (hyb_coeff[2] + middle_mat[:, 2]) * b88_grids[1][1]
        vsigma += (hyb_coeff[3] + middle_mat[:, 3]) * lyp_grids[1][1]

        evxc_real = pyscf.dft.libxc.eval_xc("b3lyp", rho, 0)
        print(evxc_real[1][0] - vrho)
        print(evxc_real[1][1] - vsigma)
        print(rho[0] * evxc_real[0] * grids.weights - excsum * grids.weights)

        vxc = (vrho, vsigma, None, None)

        vmat = np.zeros((ks.mol.nao, ks.mol.nao))
        aow = None

        if ks.mol.spin == 0:
            ao = ni.eval_ao(ks.mol, grids.coords, deriv=2)
            aow = np.ndarray(ao[0].shape, order="F", buffer=aow)

            den = rho[0] * grids.weights
            nelec = den.sum()
            excsum += np.dot(den, exc_b3lyp)

            wv = _rks_gga_wv0(rho, vxc, grids.weights)
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv, out=aow)
            aow = _scale_ao(ao[:4], wv, out=aow)
            vmat += _dot_ao_ao(ks.mol, ao[0], aow, None, None, None)

            # # FIXME: .5 * .5   First 0.5 for v+v.T symmetrization.
            # # Second 0.5 is due to the Libxc convention tau = 1/2 \nabla\phi\dot\nabla\phi
            # wv = (0.5 * 0.5 * grids.weights * vtau).reshape(-1, 1)
            # vmat += _dot_ao_ao(ks.mol, ao[1], wv * ao[1], None, None, None)
            # vmat += _dot_ao_ao(ks.mol, ao[2], wv * ao[2], None, None, None)
            # vmat += _dot_ao_ao(ks.mol, ao[3], wv * ao[3], None, None, None)

            rho = exc = vxc = vrho = vsigma = wv = None

        return nelec, excsum, vmat + vmat.T
