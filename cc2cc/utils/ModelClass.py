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
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=args.lr,
                # weight_decay=0.1,
            )
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=50,
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=args.lr,
                # weight_decay=0.1,
            )
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

    def tot_loss(self, loss_ene, loss_ene_abs, data_weight=1):
        """
        Calculate the total loss.
        """
        tot_loss = loss_ene / np.sqrt(data_weight)
        tot_loss += loss_ene_abs * self.loss_multiplier * np.sqrt(data_weight)
        return tot_loss

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
        if database_train.if_load_to_gpu_once:
            database_train.shuffle()

        for batch in database_train.data_gpu:
            if not database_train.if_load_to_gpu_once:
                batch = database_train.process_batch(batch)
            data_weight = database_train.data_weight[batch["name"]]

            with torch.autocast(device_type="cuda", dtype=self.dtype):
                loss_ene, loss_ene_abs = self.loss(batch)
                tot_loss = (
                    self.tot_loss(loss_ene, loss_ene_abs, data_weight)
                    / self.iters_to_accumulate
                )

            self.scaler.scale(tot_loss).backward()
            self.update()

            number_batch_name = len(batch["weight"])
            loss_ene_name = loss_ene.item()
            loss_ene_abs_name = loss_ene_abs.item()
            loss_tot_name = tot_loss.item()

            name_l.append(batch["name"])
            if isinstance(self.loss_ene, torch.nn.L1Loss):
                loss_ene_l.append(AU2KCALMOL * loss_ene_name)
                loss_ene_abs_l.append(AU2KCALMOL * loss_ene_abs_name)
                loss_tot_l.append(AU2KCALMOL * loss_tot_name)
            elif isinstance(self.loss_ene, torch.nn.MSELoss):
                loss_ene_l.append(AU2KCALMOL * np.sqrt(loss_ene_name))
                loss_ene_abs_l.append(
                    AU2KCALMOL * np.sqrt(loss_ene_abs_name * number_batch_name)
                )
                loss_tot_l.append(AU2KCALMOL * np.sqrt(loss_tot_name))
            else:
                raise ValueError("Unknown loss function")

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
            loss_tot_name = self.tot_loss(loss_ene, loss_ene_abs, data_weight=1).item()

            name_l.append(batch["name"])
            if isinstance(self.loss_ene, torch.nn.L1Loss):
                loss_ene_l.append(AU2KCALMOL * loss_ene_name)
                loss_ene_abs_l.append(AU2KCALMOL * loss_ene_abs_name)
                loss_tot_l.append(AU2KCALMOL * loss_tot_name)
            elif isinstance(self.loss_ene, torch.nn.MSELoss):
                loss_ene_l.append(AU2KCALMOL * np.sqrt(loss_ene_name))
                loss_ene_abs_l.append(
                    AU2KCALMOL * np.sqrt(loss_ene_abs_name * number_batch_name)
                )
                loss_tot_l.append(AU2KCALMOL * np.sqrt(loss_tot_name))
            else:
                raise ValueError("Unknown loss function")

        return (
            name_l,
            np.array(loss_ene_l),
            np.array(loss_ene_abs_l),
            np.array(loss_tot_l),
        )
