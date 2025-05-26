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

from cc2cc.utils.env_var import CHECKPOINTS_PATH
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.model.densenet import Model as ModelDensenet
from cc2cc.utils.model.transformer import Model as ModelTransformer
from cc2cc.utils.model.transformer_c_ang import Model as ModelTransformer_c_Ang
from cc2cc.utils.model.transformer_c_ang_slice import (
    Model as ModelTransformer_c_Ang_slice,
)
from cc2cc.utils.model.densenet_c import Model as ModelDensenet_c
from cc2cc.utils.model.transformer_7 import Model as ModelTransformer_7


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

        if args.model == "densenet":
            Model = ModelDensenet
            print("Model: Densenet")
        elif args.model == "transformer":
            Model = ModelTransformer
            print("Model: Transformer")
        elif args.model == "transformer_7":
            Model = ModelTransformer_7
            print("Model: Transformer_7")
        elif args.model == "densenet_c":
            Model = ModelDensenet_c
            print("Model: Densenet_c")
        elif args.model == "transformer_c_ang":
            Model = ModelTransformer_c_Ang
            print("Model: Transformer_c_Ang")
        elif args.model == "transformer_c_ang_slice":
            Model = ModelTransformer_c_Ang_slice
            print("Model: Transformer_c_Ang_slice")
        else:
            raise ValueError("Unknown model")

        self.load = getattr(args, "load", "")
        self.with_eval = getattr(args, "with_eval", True)
        self.load_epoch = getattr(args, "load_epoch", -1)
        self.save_dir = getattr(args, "save_dir", "")
        self.basis = getattr(args, "basis", "cc-pVDZ")
        self.iters_to_accumulate = getattr(args, "iters_to_accumulate", 1)
        self.max_norm = getattr(args, "max_norm", -1)
        self.weight_decay = getattr(args, "weight_decay", 1e-3)
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
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=self.weight_decay,
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
                weight_decay=self.weight_decay,
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.eval_step * 50,
                eta_min=args.lr / 100,
            )

        self.scaler = GradScaler("cuda")

        self.loss_multiplier = args.loss_multiplier
        # self.loss_ene = torch.nn.L1Loss(reduction="sum")
        self.loss_ene = torch.nn.MSELoss(reduction="sum")
        # self.loss_ene_abs = torch.nn.L1Loss(reduction="sum")
        self.loss_ene_abs = torch.nn.MSELoss(reduction="sum")

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

    def tot_loss(self, loss_ene, loss_ene_abs):
        """
        Calculate the total loss.
        """
        if self.loss_multiplier > 1e-6:
            if isinstance(self.loss_ene, torch.nn.L1Loss):
                tot_loss = loss_ene + loss_ene_abs * self.loss_multiplier
            elif isinstance(self.loss_ene, torch.nn.MSELoss):
                tot_loss = loss_ene + loss_ene_abs * self.loss_multiplier**2
            else:
                raise ValueError("Unknown loss function")
        else:
            tot_loss = loss_ene
        return tot_loss

    def loss(self, batch, data_weight):
        """
        Calculate the loss.
        """
        input_mat = batch["input"]
        weight = batch["weight"]
        output_mat_real = batch["output"] * weight
        data_weight = np.sqrt(data_weight)

        output_mat = self.model(input_mat) * weight

        loss_ene = self.loss_ene(
            data_weight * torch.sum(output_mat_real),
            data_weight * torch.sum(output_mat),
        )
        loss_ene_abs = self.loss_ene_abs(
            data_weight * output_mat_real,
            data_weight * output_mat,
        )
        tot_loss = self.tot_loss(loss_ene, loss_ene_abs) / self.iters_to_accumulate

        loss_record = np.abs(torch.sum(output_mat_real - output_mat).item())
        loss_abs_record = np.abs(
            torch.sum(torch.abs(output_mat_real - output_mat)).item()
        )

        return tot_loss, loss_record, loss_abs_record

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
        name_l, loss_ene_l, loss_ene_abs_l, loss_tot_l = [], [], [], []
        database_train.shuffle()

        for name in database_train.name_list:
            batch = database_train.data_gpu[name]
            data_weight = database_train.data_weight[name]

            if not database_train.if_load_to_gpu_once:
                batch = database_train.process_batch(batch)

            with torch.autocast(device_type="cuda", dtype=self.dtype):
                tot_loss, loss_record, loss_abs_record = self.loss(batch, data_weight)

            self.scaler.scale(tot_loss).backward()
            self.update()

            loss_tot_record = tot_loss.item()
            name_l.append(name)
            loss_ene_l.append(AU2KCALMOL * loss_record)
            loss_ene_abs_l.append(AU2KCALMOL * loss_abs_record)
            loss_tot_l.append(AU2KCALMOL * loss_tot_record)

        return (
            name_l,
            np.array(loss_ene_l),
            np.array(loss_ene_abs_l),
            np.array(loss_tot_l),
        )

    def eval_model(self, database_eval):
        """
        Evaluate the model.
        """
        self.eval()
        name_l, loss_ene_l, loss_ene_abs_l, loss_tot_l = [], [], [], []

        for name in database_eval.name_list:
            batch = database_eval.data_gpu[name]
            data_weight = database_eval.data_weight[name]

            if not database_eval.if_load_to_gpu_once:
                batch = database_eval.process_batch(batch)

            with torch.autocast(device_type="cuda", dtype=self.dtype):
                with torch.no_grad():
                    tot_loss, loss_record, loss_abs_record = self.loss(
                        batch, data_weight
                    )

            loss_tot_record = tot_loss.item()
            name_l.append(name)
            loss_ene_l.append(AU2KCALMOL * loss_record)
            loss_ene_abs_l.append(AU2KCALMOL * loss_abs_record)
            loss_tot_l.append(AU2KCALMOL * loss_tot_record)

        return (
            name_l,
            np.array(loss_ene_l),
            np.array(loss_ene_abs_l),
            np.array(loss_tot_l),
        )
