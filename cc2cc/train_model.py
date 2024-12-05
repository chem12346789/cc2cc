"""Module providing a training method."""

import argparse
import os

from tqdm import trange
import torch

import numpy as np
import wandb

from cc2cc.utils import add_args, save_csv_loss
from cc2cc.utils import DataBase, ModelDict


def train_model(train_str_dict, eval_str_dict):
    """
    Train the model.
    train_str_dict: list of training molecules
    eval_str_dict: list of evaluation molecules
    Other parameter are from the argparse.
    """
    # 0. Init the criterion and the model
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    experiment = wandb.init(
        project="DFT2CC",
        resume="allow",
        name="dft2cc",
        dir="/home/chenzihao/workdir/tmp",
        allow_val_change=True,
    )
    wandb.define_metric("*", step_metric="global_step")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modeldict = ModelDict(
        args.load,
        device,
        args.precision,
        with_eval=args.with_eval,
    )
    modeldict.load_model()

    database_train = DataBase(
        train_str_dict,
        args.extend_atom,
        args.extend_xyz,
        args.distance_list,
        args.basis,
        args.batch_size,
        device,
        args.precision,
    )
    database_eval = DataBase(
        eval_str_dict,
        args.extend_atom,
        args.extend_xyz,
        args.distance_list,
        args.basis,
        args.batch_size,
        device,
        args.precision,
    )

    experiment_dict = {
        "batch_size": args.batch_size,
        "n_train": len(database_train.name_list),
        "n_eval": len(database_eval.name_list),
        "precision": args.precision,
        "basis": args.basis,
        "with_eval": args.with_eval,
        "load": args.load,
        "jobid": os.environ.get("SLURM_JOB_ID"),
        "checkpoint": modeldict.dir_checkpoint.stem,
    }
    print(experiment_dict)
    experiment.config.update(experiment_dict)

    print(f"Start training at {modeldict.dir_checkpoint}")
    pbar0 = trange(args.epoch + 1, mininterval=2, maxinterval=20)
    for epoch in pbar0:
        train_loss_ene, train_loss_ene_tot = modeldict.train_model(database_train)
        if not modeldict.with_eval:
            modeldict.scheduler.step()

        if epoch % args.eval_step == 0:
            eval_loss_ene, eval_loss_ene_tot = modeldict.eval_model(database_eval)
            if modeldict.with_eval:
                modeldict.scheduler.step(
                    np.mean(modeldict.tot_loss(eval_loss_ene, eval_loss_ene_tot))
                )

            experiment_dict = {
                "epoch": epoch,
                "global_step": epoch,
                "train_loss_ene": np.mean(train_loss_ene),
                "train_loss_ene_tot": np.mean(train_loss_ene_tot),
                "train_loss_tot": np.mean(
                    modeldict.tot_loss(train_loss_ene, train_loss_ene_tot)
                ),
                "eval_loss_ene": np.mean(eval_loss_ene),
                "eval_loss_ene_tot": np.mean(eval_loss_ene_tot),
                "eval_loss_tot": np.mean(
                    modeldict.tot_loss(eval_loss_ene, eval_loss_ene_tot)
                ),
                "lr": modeldict.optimizer.param_groups[0]["lr"],
            }
            experiment.log(experiment_dict)

            pbar0.set_description(
                f"Epoch: {epoch}, Ene loss tot: {experiment_dict['train_loss_ene_tot']:.2f}, Ene loss eval tot: {experiment_dict['eval_loss_ene_tot']:.2f}, Loss: {experiment_dict['train_loss_tot']:.2f}, Loss_eval: {experiment_dict['eval_loss_tot']:.2f}, lr: {experiment_dict['lr']:.2e}",
                refresh=False,
            )

        if (epoch % (args.eval_step * 50) == 0) and (epoch != 0):
            save_csv_loss(
                database_train.name_list,
                modeldict.dir_checkpoint / "loss" / f"train-loss-{epoch}",
                {
                    "train_loss_ene": train_loss_ene,
                    "train_loss_ene_tot": train_loss_ene_tot,
                },
            )
            save_csv_loss(
                database_eval.name_list,
                modeldict.dir_checkpoint / "loss" / f"eval-loss-{epoch}",
                {
                    "eval_loss_ene": eval_loss_ene,
                    "eval_loss_ene_tot": eval_loss_ene_tot,
                },
            )
            modeldict.save_model(epoch)
    pbar0.close()
