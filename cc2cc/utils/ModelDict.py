"""
Generate list of model.
"""

from pathlib import Path
import datetime
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.amp import GradScaler

import pyscf
from pyscf import lib
from pyscf.dft.numint import (
    _scale_ao,
    _dot_ao_ao,
    _scale_ao_sparse,
    _dot_ao_ao_sparse,
    _tau_dot_sparse,
)
from pyscf.dft.numint import _rks_gga_wv0, _uks_gga_wv0
from pyscf.dft.libxc import xc_type
from pyscf.dft.gen_grid import BLKSIZE, NBINS, CUTOFF, ALIGNMENT_UNIT, make_mask

from cc2cc.utils.env_var import CHECKPOINTS_PATH, TEST
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.Grids import Grid
from cc2cc.utils.model.densenet import Model as ModelDensenet
from cc2cc.utils.model.transformer import Model as ModelTransformer
from cc2cc.utils.model.transformer_4_ang import Model as ModelTransformer_4_Ang


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
        self.model_name = args.model

        if args.model == "densenet":
            Model = ModelDensenet
            print("Model: Densenet")
        elif args.model == "transformer":
            Model = ModelTransformer
            print("Model: Transformer")
        elif args.model == "transformer_4_ang":
            Model = ModelTransformer_4_Ang
            print("Model: Transformer_4_Ang")
        else:
            raise ValueError("Unknown model")

        self.load = args.load
        self.with_eval = args.with_eval
        self.load_epoch = args.load_epoch
        self.save_dir = args.save_dir
        self.basis = args.basis
        self.iters_to_accumulate = args.iters_to_accumulate
        self.max_norm = args.max_norm
        self.device = torch.device(args.device)
        self.dtype = torch.float32
        self.update_counter = 0

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
        self.load_model()

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
                T_max=args.eval_step * 50,
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
                        data_loss_train = pd.read_csv(path)["train_loss_ene"]
                        data_loss_eval = pd.read_csv(
                            load_checkpoint / "loss" / f"eval-loss-{load_epoch}"
                        )["train_loss_ene"]
                        mean_loss = np.mean(np.append(data_loss_train, data_loss_eval))
                        print(f"Mean Loss: {mean_loss} of epoch {load_epoch}")
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

    def update(self):
        """
        Update the model.

        # See https://kozodoi.me/blog/20210219/gradient-accumulation and
        # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
        """
        self.update_counter += 1

        if self.update_counter % self.iters_to_accumulate != 0:
            return

        if self.max_norm != -1:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.zero_grad()
        self.update_counter = 0

    def loss(self, batch):
        """
        Calculate the loss.
        """
        input_mat = batch["input"]
        weight = batch["weight"]
        output_mat_real = batch["output"] * weight

        output_mat = self.model(input_mat) * weight

        loss_ene = self.loss_ene(
            torch.sum(output_mat_real),
            torch.sum(output_mat),
        )

        loss_ene_abs = self.loss_ene_abs(
            output_mat_real,
            output_mat,
        )

        return loss_ene, loss_ene_abs

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
        1 / self.iters_to_accumulate is the effective batch size.
        See https://kozodoi.me/blog/20210219/gradient-accumulation and
        https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
        """
        self.train()
        name_l, loss_ene_l, loss_ene_abs_l = [], [], []
        if database_train.if_load_to_gpu_once:
            database_train.shuffle()

        for batch in database_train.data_gpu:
            if not database_train.if_load_to_gpu_once:
                batch = database_train.process_batch(batch)
            data_weight = database_train.data_weight[batch["name"]]

            with torch.autocast(device_type="cuda", dtype=self.dtype):
                loss_ene, loss_ene_abs = self.loss(batch)
                tot_loss = (
                    self.tot_loss(loss_ene, loss_ene_abs)
                    * data_weight
                    / self.iters_to_accumulate
                )

            self.scaler.scale(tot_loss).backward()
            self.update()

            number_batch_name = len(batch["weight"])
            loss_ene_name = loss_ene.item()
            loss_ene_abs_name = loss_ene_abs.item()

            name_l.append(batch["name"])
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

        return name_l, np.array(loss_ene_l), np.array(loss_ene_abs_l)

    def eval_model(self, database_eval):
        """
        Evaluate the model.
        """
        self.eval()
        name_l, loss_ene_l, loss_ene_abs_l = [], [], []
        if database_eval.if_load_to_gpu_once:
            database_eval.shuffle()

        for batch in database_eval.data_gpu:
            if not database_eval.if_load_to_gpu_once:
                batch = database_eval.process_batch(batch)

            with torch.no_grad():
                loss_ene, loss_ene_abs = self.loss(batch)
            number_batch_name = len(batch["weight"])
            loss_ene_name = loss_ene.item()
            loss_ene_abs_name = loss_ene_abs.item()

            name_l.append(batch["name"])
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

        return name_l, np.array(loss_ene_l), np.array(loss_ene_abs_l)

    def get_nev_4(
        self,
        ni,
        mol,
        grids: Grid,
        dms,
        xc_code="b3lyp",
    ):
        """
        Obtain the energy density.
        Input: dft instance and grids instance.
        Output: the potential (ngrids).
        """
        ao = ni.eval_ao(mol, grids.coords, deriv=1)
        if mol.spin == 0:
            rho = ni.eval_rho(mol, ao, dms, xctype=xc_type(xc_code))
        else:
            rho = [
                ni.eval_rho(mol, ao, dms[0], xctype=xc_type(xc_code)),
                ni.eval_rho(mol, ao, dms[1], xctype=xc_type(xc_code)),
            ]

        rho_cube = grids.gen_4(mol, dms, reset=True)
        input_mat = torch.tensor(rho_cube, dtype=self.dtype, device=self.device)
        input_mat.requires_grad = True
        output_mat = self.model(input_mat)[:, 0]

        middle_cube = torch.autograd.grad(
            torch.sum(output_mat),
            input_mat,
            create_graph=True,
        )[0]
        middle_mat = middle_cube.detach().cpu().numpy()
        energy_den = output_mat.detach().cpu().numpy()

        return rho, energy_den, middle_mat

    def get_nev_cube(
        self,
        ni,
        mol,
        grids: Grid,
        dms,
        xc_code="b3lyp",
    ):
        """
        Obtain the energy density.
        Input: dft instance and grids instance.
        Output: the potential (ngrids).
        """
        time_start = time.time()

        ao = ni.eval_ao(mol, grids.coords, deriv=1)
        if mol.spin == 0:
            rho = ni.eval_rho(mol, ao, dms, xctype=xc_type(xc_code))
        else:
            rho = [
                ni.eval_rho(mol, ao, dms[0], xctype=xc_type(xc_code)),
                ni.eval_rho(mol, ao, dms[1], xctype=xc_type(xc_code)),
            ]

        time_rho = time.time()
        print(f"    Time for rho: {time_rho - time_start:.2f} seconds")

        rho_cube = grids.gen_cube_rho(mol, dms, reset=True)

        time_rho_cube = time.time()
        print(f"    Time for rho_cube: {time_rho_cube - time_rho:.2f} seconds")

        input_mat = torch.tensor(
            rho_cube,
            dtype=self.dtype,
            device=self.device,
        )
        input_mat.requires_grad = True
        output_mat = self.model(input_mat)[:, 0]

        time_output_mat = time.time()
        print(f"    Time for output_mat: {time_output_mat - time_rho_cube:.2f} seconds")

        middle_cube = torch.autograd.grad(
            torch.sum(output_mat),
            input_mat,
            create_graph=True,
        )[0]

        time_middle_cube = time.time()
        print(
            f"    Time for middle_cube: {time_middle_cube - time_output_mat:.2f} seconds"
        )

        middle_mat = grids.get_center_density(middle_cube).detach().cpu().numpy()
        energy_den = output_mat.detach().cpu().numpy()

        time_end = time.time()
        print(f"    Time for output_mat: {time_end - time_middle_cube:.2f} seconds")

        return rho, energy_den, middle_mat

    def get_nev(
        self,
        ni,
        mol,
        grids: Grid,
        dms,
        xc_code="b3lyp",
        hermi=1,
        max_memory=2000,
    ):
        """
        Obtain the nelec, excsum, and vmat.
        """
        time_start = time.time()

        if self.model_name == "transformer_4_ang":
            rho, energy_den, middle_mat = self.get_nev_4(ni, mol, grids, dms, xc_code)
        else:
            rho, energy_den, middle_mat = self.get_nev_cube(
                ni, mol, grids, dms, xc_code
            )

        time_get_nev = time.time()
        print(
            f"Time for get_nev (rho, energy_den, middle_mat, middle_cube, get_nev_cube): {time_get_nev - time_start:.2f} seconds"
        )

        if TEST:
            hyb_coeff = [0.0, 0.0, 0.0, 0.0]
        else:
            hyb_coeff = [0.08, 0.19, 0.72, 0.81]

        if mol.spin == 0:
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

        time_mix_b3lyp = time.time()
        print(f"Time for mix_b3lyp: {time_mix_b3lyp - time_get_nev:.2f} seconds")

        exc_b3lyp = (
            hyb_coeff[0] * lda_grids[0]
            + hyb_coeff[1] * vwn_grids[0]
            + hyb_coeff[2] * b88_grids[0]
            + hyb_coeff[3] * lyp_grids[0]
        )

        vxc = (vrho, vsigma, None, None)
        aow = None
        ao = ni.eval_ao(mol, grids.coords, deriv=1)
        aow = np.ndarray(ao[0].shape, order="F", buffer=aow)

        if mol.spin == 0:
            vmat = np.zeros((mol.nao, mol.nao))
            den = rho[0] * grids.weights
            nelec = den.sum()
            excsum = (energy_den * grids.weights).sum()
            excsum += np.dot(den, exc_b3lyp)

            wv = _rks_gga_wv0(rho, vxc, grids.weights)
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv, out=aow)
            aow = _scale_ao(ao[:4], wv, out=aow)
            vmat += _dot_ao_ao(mol, ao[0], aow, None, None, None)

            rho = vxc = vrho = vsigma = wv = None

            vmat = vmat + vmat.T
        else:
            vmat = np.zeros((2, mol.nao, mol.nao))
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
            vmat[0] += _dot_ao_ao(mol, ao[0], aow, None, None, None)
            #:aow = numpy.einsum('npi,np->pi', ao, wvb, out=aow)
            aow = _scale_ao(ao, wvb, out=aow)
            vmat[1] += _dot_ao_ao(mol, ao[0], aow, None, None, None)

            rho = vxc = wva = wvb = None
            vmat[0] = vmat[0] + vmat[0].T
            vmat[1] = vmat[1] + vmat[1].T

        time_end = time.time()
        print(f"Time for get_fock: {time_end - time_mix_b3lyp:.2f} seconds")

        return nelec, excsum, vmat

    def nr_rks(
        self,
        ni,
        mol,
        grids: Grid,
        dms,
        xc_code="b3lyp",
        hermi=1,
        max_memory=2000,
        verbose=None,
    ):
        """
        Obtain the nelec, excsum, and vmat.
        """
        xctype = ni._xc_type(xc_code)
        make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
        ao_loc = mol.ao_loc_nr()
        cutoff = grids.cutoff * 1e2
        nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))

        nelec = np.zeros(nset)
        excsum = np.zeros(nset)
        vmat = np.zeros((nset, nao, nao))

        def block_loop(ao_deriv):
            for ao, mask, weight, coords in ni.block_loop(
                mol, grids, nao, ao_deriv, max_memory=max_memory
            ):
                for i in range(nset):
                    rho = make_rho(i, ao, mask, xctype)
                    exc_lda, vxc_lda = ni.eval_xc_eff(
                        "LDA,", rho[0], deriv=1, xctype=ni._xc_type("LDA,")
                    )[:2]
                    exc_vwn, vxc_vwn = ni.eval_xc_eff(
                        ",VWN3", rho[0], deriv=1, xctype=ni._xc_type(",VWN3")
                    )[:2]
                    exc_b88, vxc_b88 = ni.eval_xc_eff(
                        "B88,", rho, deriv=1, xctype=ni._xc_type("B88,")
                    )[:2]
                    exc_lyp, vxc_lyp = ni.eval_xc_eff(
                        ",LYP", rho, deriv=1, xctype=ni._xc_type(",LYP")
                    )[:2]
                    exc = 0.72 * exc_b88 + 0.81 * exc_lyp
                    vxc = 0.72 * vxc_b88 + 0.81 * vxc_lyp
                    exc += 0.08 * exc_lda + 0.19 * exc_vwn
                    vxc[[0], :] += 0.08 * vxc_lda + 0.19 * vxc_vwn
                    # exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[:2]
                    if xctype == "LDA":
                        den = rho * weight
                    else:
                        den = rho[0] * weight
                    nelec[i] += den.sum()
                    excsum[i] += np.dot(den, exc)
                    wv = weight * vxc
                    yield i, ao, mask, wv

        aow = None
        pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)

        if xctype == "LDA":
            ao_deriv = 0
            for i, ao, mask, wv in block_loop(ao_deriv):
                _dot_ao_ao_sparse(
                    ao, ao, wv, nbins, mask, pair_mask, ao_loc, hermi, vmat[i]
                )

        elif xctype == "GGA":
            ao_deriv = 1
            for i, ao, mask, wv in block_loop(ao_deriv):
                wv[0] *= 0.5  # *.5 because vmat + vmat.T at the end
                aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
                _dot_ao_ao_sparse(
                    ao[0],
                    aow,
                    None,
                    nbins,
                    mask,
                    pair_mask,
                    ao_loc,
                    hermi=0,
                    out=vmat[i],
                )
            vmat = lib.hermi_sum(vmat, axes=(0, 2, 1))

        elif xctype == "MGGA":
            if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
                raise NotImplementedError("laplacian in meta-GGA method")
            ao_deriv = 1
            v1 = np.zeros_like(vmat)
            for i, ao, mask, wv in block_loop(ao_deriv):
                wv[0] *= 0.5  # *.5 for v+v.conj().T
                wv[4] *= 0.5  # *.5 for 1/2 in tau
                aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
                _dot_ao_ao_sparse(
                    ao[0],
                    aow,
                    None,
                    nbins,
                    mask,
                    pair_mask,
                    ao_loc,
                    hermi=0,
                    out=vmat[i],
                )
                _tau_dot_sparse(
                    ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1[i]
                )
            vmat = lib.hermi_sum(vmat, axes=(0, 2, 1))
            vmat += v1

        elif xctype == "HF":
            pass
        else:
            raise NotImplementedError(f"numint.nr_uks for functional {xc_code}")

        if nset == 1:
            nelec = nelec[0]
            excsum = excsum[0]
            vmat = vmat[0]

        if isinstance(dms, np.ndarray):
            dtype = dms.dtype
        else:
            dtype = np.result_type(*dms)
        if vmat.dtype != dtype:
            vmat = np.asarray(vmat, dtype=dtype)
        return nelec, excsum, vmat
