"""
Generate list of model.
"""

from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.amp import GradScaler

import pyscf
from pyscf.dft.numint import _scale_ao, _dot_ao_ao
from pyscf.dft.numint import _rks_gga_wv0, _uks_gga_wv0
from pyscf.dft.libxc import xc_type

from cc2cc.utils.env_var import CHECKPOINTS_PATH, TEST
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.Grids import Grid
from cc2cc.utils.model.densenet import Model as ModelDensenet
from cc2cc.utils.model.transformer import Model as ModelTransformer


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
        if args.model == "densenet":
            Model = ModelDensenet
            print("Model: Densenet")
        elif args.model == "transformer":
            Model = ModelTransformer
            print("Model: Transformer")
        else:
            raise ValueError("Unknown model")

        self.load = args.load
        self.with_eval = args.with_eval
        self.load_epoch = args.load_epoch
        self.save_dir = args.save_dir
        self.basis = args.basis
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        if self.save_dir is not None and self.save_dir != "":
            self.dir_checkpoint = (
                CHECKPOINTS_PATH / f"checkpoint-ccdft_{self.basis}_{self.save_dir}"
            ).resolve()
            if not self.dir_checkpoint.exists():
                print(f"Directory {self.dir_checkpoint} not found. Created!")
                (self.dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)
        else:
            self.dir_checkpoint = (
                CHECKPOINTS_PATH
                / f"checkpoint-ccdft_{self.basis}_{datetime.datetime.today():%Y-%m-%d-%H-%M-%S}/"
            ).resolve()

        self.model: torch.nn.Module = Model().to(self.device)
        if args.precision == "float64":
            self.dtype = torch.float64
            self.model.double()

        if self.with_eval:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=50,
            )
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=250,
                eta_min=args.lr / 100,
            )

        self.scaler = GradScaler("cuda")

        self.loss_multiplier = args.loss_multiplier
        self.loss_ene = torch.nn.L1Loss(reduction="sum")
        # self.loss_ene = torch.nn.MSELoss(reduction="sum")
        self.loss_ene_abs = torch.nn.L1Loss(reduction="sum")
        # self.loss_ene_abs = torch.nn.MSELoss(reduction="sum")

    def load_model(self):
        """
        Load the model from the checkpoint.
        """
        load_checkpoint = Path(
            CHECKPOINTS_PATH / f"checkpoint-ccdft_{self.basis}_{self.load}/"
        ).resolve()
        list_of_path = list(load_checkpoint.glob("*.pth"))
        if len(list_of_path) == 0:
            print(f"No model found in {load_checkpoint}, use random initialization.")
        else:
            if self.load_epoch < 0:
                min_loss = None
                if (load_checkpoint / "loss").exists():
                    for path in list((load_checkpoint / "loss").glob("train-loss-*")):
                        load_epoch = path.stem.split("-")[-1]
                        if abs(int(load_epoch)) < abs(self.load_epoch):
                            continue
                        data_loss = pd.read_csv(path)
                        mean_loss = np.mean(data_loss["train_loss_ene"])
                        data_loss = pd.read_csv(
                            load_checkpoint / "loss" / f"eval-loss-{load_epoch}"
                        )
                        mean_loss += np.mean(data_loss["train_loss_ene"])
                        print(mean_loss)
                        if min_loss is None or mean_loss < min_loss:
                            min_loss = mean_loss
                            load_path = load_checkpoint / f"{load_epoch}.pth"
                else:
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

    def update(self, idx, max_norm=-1, step=-1):
        """
        Update the model.
        """
        if max_norm != -1:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if step == -1:
            self.zero_grad()
        else:
            if (idx + 1) % step == 0:
                self.zero_grad()

    def loss(self, batch, **kwargs):
        """
        Calculate the loss.
        """
        input_mat = batch["input"]
        weight = batch["weight"]
        output_mat_real = batch["output"]
        output_mat = self.model(input_mat)

        loss_ene = self.loss_ene(
            torch.sum(output_mat_real * weight),
            torch.sum(output_mat * weight),
        )

        loss_ene_abs = self.loss_ene_abs(
            output_mat_real * weight,
            output_mat * weight,
        )

        data_weight = kwargs.get("data_weight", None)
        if data_weight is None:
            return loss_ene, loss_ene_abs
        else:
            return (data_weight * loss_ene, data_weight * loss_ene_abs)

    def tot_loss(self, loss_ene, loss_ene_abs):
        """
        Calculate the total loss.
        """
        return loss_ene + self.loss_multiplier * loss_ene_abs

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
        loss_ene_l, loss_ene_abs_l = [], []
        database_train.rng.shuffle(database_train.name_list)

        for name in database_train.name_list:
            loss_ene_name = 0.0
            number_batch_name = 0
            loss_ene_abs_name = 0.0

            for idx, batch in enumerate(database_train.data_gpu[name]):

                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    loss_ene, loss_ene_abs = self.loss(
                        batch,
                        data_weight=database_train.data_weight[name],
                    )

                tot_loss = self.tot_loss(loss_ene, loss_ene_abs)
                self.scaler.scale(tot_loss).backward()

                self.update(idx, 0.5, 25)

                number_batch_name += len(batch["weight"])
                loss_ene_name += loss_ene.item()
                loss_ene_abs_name += loss_ene_abs.item()

            if isinstance(self.loss_ene, torch.nn.L1Loss):
                loss_ene_l.append(AU2KCALMOL * loss_ene_name)
            elif isinstance(self.loss_ene, torch.nn.MSELoss):
                loss_ene_l.append(AU2KCALMOL * np.sqrt(loss_ene_name))
            else:
                raise ValueError("Unknown loss function")

            if isinstance(self.loss_ene_abs, torch.nn.L1Loss):
                loss_ene_abs_l.append(AU2KCALMOL * loss_ene_abs_name)
            elif isinstance(self.loss_ene_abs, torch.nn.MSELoss):
                loss_ene_abs_l.append(
                    AU2KCALMOL * np.sqrt(loss_ene_abs_name * number_batch_name)
                )
            else:
                raise ValueError("Unknown loss function")

        return np.array(loss_ene_l), np.array(loss_ene_abs_l)

    def eval_model(self, database_eval):
        """
        Evaluate the model.
        """
        self.eval()
        loss_ene_l, loss_ene_abs_l = [], []

        for name in database_eval.name_list:
            loss_ene_name = 0.0
            number_batch_name = 0
            loss_ene_abs_name = 0.0

            for batch in database_eval.data_gpu[name]:
                with torch.no_grad():
                    loss_ene, loss_ene_abs = self.loss(
                        batch,
                        data_weight=database_eval.data_weight[name],
                    )
                number_batch_name += len(batch["weight"])
                loss_ene_name += loss_ene.item()
                loss_ene_abs_name += loss_ene_abs.item()

            if isinstance(self.loss_ene, torch.nn.L1Loss):
                loss_ene_l.append(AU2KCALMOL * loss_ene_name)
            elif isinstance(self.loss_ene, torch.nn.MSELoss):
                loss_ene_l.append(AU2KCALMOL * np.sqrt(loss_ene_name))
            else:
                raise ValueError("Unknown loss function")

            if isinstance(self.loss_ene_abs, torch.nn.L1Loss):
                loss_ene_abs_l.append(AU2KCALMOL * loss_ene_abs_name)
            elif isinstance(self.loss_ene_abs, torch.nn.MSELoss):
                loss_ene_abs_l.append(
                    AU2KCALMOL * np.sqrt(loss_ene_abs_name * number_batch_name)
                )
            else:
                raise ValueError("Unknown loss function")

        return np.array(loss_ene_l), np.array(loss_ene_abs_l)

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
        ao = ni.eval_ao(ks.mol, grids.coords, deriv=1)
        if ks.mol.spin == 0:
            rho = ni.eval_rho(ks.mol, ao, dms, xctype=xc_type(xc_code))
        else:
            rho = [
                ni.eval_rho(ks.mol, ao, dms[0], xctype=xc_type(xc_code)),
                ni.eval_rho(ks.mol, ao, dms[1], xctype=xc_type(xc_code)),
            ]

        rho_cube = grids.gen_cube_rho(ks.mol, dms, reset=True)
        # mask_cube = np.zeros((1, 4, 3, 3, 3))
        # mask_cube[:, :, 1, 1, 1] = 1.0
        # print(rho_cube[:10])
        # rho_cube = rho_cube * mask_cube
        # print(rho_cube[:10])
        input_mat = torch.tensor(rho_cube, dtype=self.dtype, device=self.device)
        input_mat.requires_grad = True
        output_mat = self.model(input_mat)[:, 0]

        middle_cube = torch.autograd.grad(
            torch.sum(output_mat),
            input_mat,
            create_graph=True,
        )[0]
        middle_mat = grids.get_center_density(middle_cube).detach().cpu().numpy()
        energy_den = output_mat.detach().cpu().numpy()

        if TEST:
            hyb_coeff = [0.0, 0.0, 0.0, 0.0]
        else:
            hyb_coeff = [0.08, 0.19, 0.72, 0.81]

        if ks.mol.spin == 0:
            rho_lda = rho[0]
            lda_grids = pyscf.dft.libxc.eval_xc("LDA,", rho_lda, 0)
            vwn_grids = pyscf.dft.libxc.eval_xc(",VWN3", rho_lda, 0)
            b88_grids = pyscf.dft.libxc.eval_xc("B88,", rho, 0)
            lyp_grids = pyscf.dft.libxc.eval_xc(",LYP", rho, 0)
            vrho = (hyb_coeff[0] + middle_mat[:, 0]) * lda_grids[1][0]
            vrho += (hyb_coeff[1] + middle_mat[:, 1]) * vwn_grids[1][0]
            vrho += (hyb_coeff[2] + middle_mat[:, 2]) * b88_grids[1][0]
            vrho += (hyb_coeff[3] + middle_mat[:, 3]) * lyp_grids[1][0]
            vsigma = (hyb_coeff[2] + middle_mat[:, 2]) * b88_grids[1][1]
            vsigma += (hyb_coeff[3] + middle_mat[:, 3]) * lyp_grids[1][1]
        else:
            rho_lda = [rho[0][0], rho[1][0]]
            lda_grids = pyscf.dft.libxc.eval_xc("LDA,", rho_lda, 1)
            vwn_grids = pyscf.dft.libxc.eval_xc(",VWN3", rho_lda, 1)
            b88_grids = pyscf.dft.libxc.eval_xc("B88,", rho, 1)
            lyp_grids = pyscf.dft.libxc.eval_xc(",LYP", rho, 1)
            vrho = (hyb_coeff[0] + middle_mat[:, 0]).reshape(-1, 1) * lda_grids[1][0]
            vrho += (hyb_coeff[1] + middle_mat[:, 1]).reshape(-1, 1) * vwn_grids[1][0]
            vrho += (hyb_coeff[2] + middle_mat[:, 2]).reshape(-1, 1) * b88_grids[1][0]
            vrho += (hyb_coeff[3] + middle_mat[:, 3]).reshape(-1, 1) * lyp_grids[1][0]
            vsigma = (hyb_coeff[2] + middle_mat[:, 2]).reshape(-1, 1) * b88_grids[1][1]
            vsigma += (hyb_coeff[3] + middle_mat[:, 3]).reshape(-1, 1) * lyp_grids[1][1]

        exc_b3lyp = (
            hyb_coeff[0] * lda_grids[0]
            + hyb_coeff[1] * vwn_grids[0]
            + hyb_coeff[2] * b88_grids[0]
            + hyb_coeff[3] * lyp_grids[0]
        )

        vxc = (vrho, vsigma, None, None)
        aow = None
        ao = ni.eval_ao(ks.mol, grids.coords, deriv=1)
        aow = np.ndarray(ao[0].shape, order="F", buffer=aow)

        if ks.mol.spin == 0:
            vmat = np.zeros((ks.mol.nao, ks.mol.nao))
            den = rho[0] * grids.weights
            nelec = den.sum()
            excsum = (energy_den * grids.weights).sum()
            excsum += np.dot(den, exc_b3lyp)

            wv = _rks_gga_wv0(rho, vxc, grids.weights)
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv, out=aow)
            aow = _scale_ao(ao[:4], wv, out=aow)
            vmat += _dot_ao_ao(ks.mol, ao[0], aow, None, None, None)

            rho = vxc = vrho = vsigma = wv = None

            vmat = vmat + vmat.T
        else:
            vmat = np.zeros((2, ks.mol.nao, ks.mol.nao))
            den = [
                rho[0][0] * grids.weights,
                rho[1][0] * grids.weights,
            ]
            nelec = [den[0].sum(), den[1].sum()]
            excsum = (energy_den * grids.weights).sum()
            excsum += np.dot(den[0], exc_b3lyp) + np.dot(den[1], exc_b3lyp)

            wva, wvb = _uks_gga_wv0((rho[0], rho[1]), vxc, grids.weights)
            # :aow = numpy.einsum('npi,np->pi', ao, wva, out=aow)
            aow = _scale_ao(ao, wva, out=aow)
            vmat[0] += _dot_ao_ao(ks.mol, ao[0], aow, None, None, None)
            #:aow = numpy.einsum('npi,np->pi', ao, wvb, out=aow)
            aow = _scale_ao(ao, wvb, out=aow)
            vmat[1] += _dot_ao_ao(ks.mol, ao[0], aow, None, None, None)

            rho = vxc = wva = wvb = None
            vmat[0] = vmat[0] + vmat[0].T
            vmat[1] = vmat[1] + vmat[1].T

        return nelec, excsum, vmat
