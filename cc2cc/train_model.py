"""Module providing a training method."""

import os
import random
import time
import numpy as np
import torch
from torchinfo import summary

import wandb

from cc2cc.utils import DataRecord
from cc2cc.utils import DataBase, ModelClass, DataBase_4

seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


def train_model(train_str_dict, eval_str_dict, args):
    """
    Train the model.
    train_str_dict: list of training molecules
    eval_str_dict: list of evaluation molecules
    Other parameter are from the argparse.
    """
    # 0. Init the criterion and the model

    experiment = wandb.init(
        project="DFT2CC",
        resume="allow",
        name="dft2cc",
        dir="~/raid/tmp",
        allow_val_change=True,
    )
    wandb.define_metric("*", step_metric="global_step")

    modeldict = ModelClass(args)

    if args.model == "transformer_4_ang":
        print(
            summary(
                modeldict.model,
                input_size=(302 * 75 * 5, 4),
                depth=10,
                dtypes=(
                    [torch.float32] if args.precision == "float32" else [torch.float64]
                ),
                mode="train",
            )
        )
        database_eval = DataBase_4(eval_str_dict, args)
        database_train = DataBase_4(train_str_dict, args)
    else:
        print(
            summary(
                modeldict.model,
                input_size=(302 * 75 * 5, 4, 3, 3, 3),
                depth=10,
                dtypes=(
                    [torch.float32] if args.precision == "float32" else [torch.float64]
                ),
                mode="train",
            )
        )
        database_eval = DataBase(eval_str_dict, args)
        database_train = DataBase(train_str_dict, args)

    experiment_dict = {
        "batch_size": args.batch_size,
        "n_train": len(database_train.data_gpu),
        "n_eval": len(database_eval.data_gpu),
        "precision": args.precision,
        "basis": args.basis,
        "with_eval": args.with_eval,
        "load": args.load,
        "jobid": os.environ.get("SLURM_JOB_ID"),
        "pid": os.getpid(),
        "checkpoint": modeldict.dir_checkpoint.stem,
        "loss_multiplier": modeldict.loss_multiplier,
        "loss_ene": (
            "L1Loss" if isinstance(modeldict.loss_ene, torch.nn.L1Loss) else "MSELoss"
        ),
        "loss_ene_abs": (
            "L1Loss"
            if isinstance(modeldict.loss_ene_abs, torch.nn.L1Loss)
            else "MSELoss"
        ),
        "iters_to_accumulate": modeldict.iters_to_accumulate,
        "max_norm": modeldict.max_norm,
    }
    print(experiment_dict)
    experiment.config.update(experiment_dict)

    print(f"Start training at {modeldict.dir_checkpoint}")
    time_start = time.time()

    for epoch in range(args.epoch + 1):
        (
            train_name_list,
            train_loss_ene,
            train_loss_ene_abs,
            train_loss_ene_tot,
        ) = modeldict.train_model(database_train)
        if not modeldict.with_eval:
            modeldict.scheduler.step()

        if epoch % args.eval_step == 0:
            (
                eval_name_list,
                eval_loss_ene,
                eval_loss_ene_abs,
                eval_loss_ene_tot,
            ) = modeldict.eval_model(database_eval)
            if modeldict.with_eval:
                modeldict.scheduler.step(np.mean(eval_loss_ene_tot))

            if epoch % (args.eval_step * 50) == 0:
                modeldict.save_model(epoch)

                data_record_train = DataRecord(
                    modeldict.dir_checkpoint / "loss" / f"train-loss-{epoch}"
                )
                data_record_train.add_data(
                    train_name_list,
                    {
                        "train_loss_ene": train_loss_ene,
                        "train_loss_ene_abs": train_loss_ene_abs,
                    },
                )
                data_record_train.save_csv()

                data_record_eval = DataRecord(
                    modeldict.dir_checkpoint / "loss" / f"eval-loss-{epoch}"
                )
                data_record_eval.add_data(
                    eval_name_list,
                    {
                        "train_loss_ene": eval_loss_ene,
                        "train_loss_ene_abs": eval_loss_ene_abs,
                    },
                )
                data_record_eval.save_csv()

                experiment_dict = {
                    "epoch_eval": epoch,
                    "train_loss_ene_epoch_eval": np.mean(train_loss_ene),
                    "eval_loss_ene_epoch_eval": np.mean(eval_loss_ene),
                    "lr": modeldict.optimizer.param_groups[0]["lr"],
                }
                experiment.log(experiment_dict)

        experiment_dict = {
            "epoch": epoch,
            "global_step": epoch,
            "train_loss_ene": np.mean(train_loss_ene),
            "train_loss_ene_abs": np.mean(train_loss_ene_abs),
            "train_loss_tot": np.mean(train_loss_ene_tot),
            "eval_loss_ene": np.mean(eval_loss_ene),
            "eval_loss_ene_abs": np.mean(eval_loss_ene_abs),
            "eval_loss_tot": np.mean(eval_loss_ene_tot),
            "lr": modeldict.optimizer.param_groups[0]["lr"],
        }
        experiment.log(experiment_dict)

        time_end = time.time()
        time_elapsed = time_end - time_start
        print(
            f"Epoch: {epoch}, "
            f"Loss: {np.mean(train_loss_ene):>5.2f}, "
            f"Eval: {np.mean(eval_loss_ene):>5.2f}, "
            f"Loss abs: {np.mean(train_loss_ene_abs):>5.2f}, "
            f"Eval abs: {np.mean(eval_loss_ene_abs):>5.2f}, "
            f"lr: {experiment_dict['lr']:>5.2e}, "
            f"Speed: {time_elapsed / (epoch + 1):>5.2f}s/epoch",
            flush=True,
        )
