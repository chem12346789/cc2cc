"""Module providing a training method."""

import os

from tqdm import trange
import numpy as np
import wandb
import torch

from cc2cc.utils import DataRecord
from cc2cc.utils import DataBase, ModelDict


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

    modeldict = ModelDict(args)
    modeldict.load_model()

    database_train = DataBase(train_str_dict, args)
    database_eval = DataBase(eval_str_dict, args)

    experiment_dict = {
        "batch_size": args.batch_size,
        "n_train": len(database_train.name_list),
        "n_eval": len(database_eval.name_list),
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
    }
    print(experiment_dict)
    experiment.config.update(experiment_dict)

    print(f"Start training at {modeldict.dir_checkpoint}")
    pbar0 = trange(args.epoch + 1, mininterval=2, maxinterval=20)
    for epoch in pbar0:
        # modeldict.loss_multiplier = args.loss_multiplier * max(
        #     min(1.0, 3 * epoch / args.epoch - 1), 0
        # )
        train_loss_ene, train_loss_ene_abs = modeldict.train_model(database_train)
        if not modeldict.with_eval:
            modeldict.scheduler.step()

        if epoch % args.eval_step == 0:
            eval_loss_ene, eval_loss_ene_abs = modeldict.eval_model(database_eval)
            if modeldict.with_eval:
                modeldict.scheduler.step(
                    np.mean(modeldict.tot_loss(eval_loss_ene, eval_loss_ene_abs))
                )

            experiment_dict = {
                "epoch": epoch,
                "global_step": epoch,
                "train_loss_ene": np.mean(train_loss_ene),
                "train_loss_ene_abs": np.mean(train_loss_ene_abs),
                "train_loss_tot": np.mean(
                    modeldict.tot_loss(train_loss_ene, train_loss_ene_abs)
                ),
                "eval_loss_ene": np.mean(eval_loss_ene),
                "eval_loss_ene_abs": np.mean(eval_loss_ene_abs),
                "eval_loss_tot": np.mean(
                    modeldict.tot_loss(eval_loss_ene, eval_loss_ene_abs)
                ),
                "lr": modeldict.optimizer.param_groups[0]["lr"],
            }
            experiment.log(experiment_dict)

            pbar0.set_description(
                f"Epoch: {epoch}, "
                f"Loss: {experiment_dict['train_loss_ene']:.2f}, "
                f"Eval: {experiment_dict['eval_loss_ene']:.2f}, "
                f"Loss abs: {experiment_dict['train_loss_ene_abs']:.2f}, "
                f"Eval abs: {experiment_dict['eval_loss_ene_abs']:.2f}, "
                f"lr: {experiment_dict['lr']:.2e}",
                refresh=False,
            )

        if epoch % 250 == 0:
            modeldict.save_model(epoch)

            data_record_train = DataRecord(
                modeldict.dir_checkpoint / "loss" / f"train-loss-{epoch}"
            )
            data_record_train.add_data(
                database_train.name_list,
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
                database_eval.name_list,
                {
                    "train_loss_ene": eval_loss_ene,
                    "train_loss_ene_abs": eval_loss_ene_abs,
                },
            )
            data_record_eval.save_csv()
    pbar0.close()
