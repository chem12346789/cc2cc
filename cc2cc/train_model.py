"""Module providing a training method."""

import os
import random
import numpy as np
import torch
from torch import distributed as dist

import wandb

from cc2cc.utils import ModelClass
from cc2cc.utils import print_computer_info
from cc2cc.utils.timer import Timer


class BestLoss:
    def __init__(self):
        self.loss_dict = {
            "tot_loss": np.inf,
            "train_loss": np.inf,
            "eval_loss": np.inf,
        }

    def update(self, now_loss):
        if_improved = False
        for key in self.loss_dict.keys():
            if now_loss[key] < self.loss_dict[key]:
                print(
                    f"Best {key} improved: {self.loss_dict[key]:.2e} -> {now_loss[key]:.2e}"
                )
                self.loss_dict[key] = now_loss[key]
                if_improved = True
        return if_improved


def train_model(train_str_dict, eval_str_dict, args):
    """
    Train the model.
    train_str_dict: list of training molecules
    eval_str_dict: list of evaluation molecules
    Other parameter are from the argparse.
    """

    # 0. Init the environment
    if args.seed is not None:
        # Set the random seed for reproducibility
        random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        print("Warning: Using deterministic mode, which may slow down training.")

    # 1. Init the criterion and the model

    modeldict = ModelClass(args)
    modeldict.init_model(init_train=True)
    modeldict.init_database(train_str_dict, eval_str_dict)

    if modeldict.args.distributed:
        dist.barrier()

    if modeldict.local_rank == 0:
        print_computer_info(args.device)

        experiment_dict = {
            "n_train": len(modeldict.database_train),
            "n_eval": len(modeldict.database_eval),
            "jobid": os.environ.get("SLURM_JOB_ID"),
            "pid": os.getpid(),
            "checkpoint": modeldict.dir_checkpoint.stem,
        }
        for key in vars(args):
            experiment_dict[key] = getattr(args, key)
        print(experiment_dict)

        run = wandb.init(
            project="DFT2CC",
            resume="allow",
            name="dft2cc",
            config=experiment_dict,
            allow_val_change=True,
        )
        wandb.define_metric("*", step_metric="global_step")

        timer = Timer()
        best_loss = BestLoss()

    if modeldict.args.distributed:
        dist.barrier()

    for epoch in range(args.epoch + 1):
        if modeldict.args.distributed:
            modeldict.database_train.sampler.set_epoch(epoch)
            modeldict.database_eval.sampler.set_epoch(epoch)

        if epoch > modeldict.start_step:
            train_record = modeldict.train_model()
            if modeldict.args.distributed:
                dist.barrier()

            if epoch % args.eval_step == 0:
                eval_record = modeldict.eval_model()
                if modeldict.args.distributed:
                    dist.barrier()

                if modeldict.local_rank == 0:
                    experiment_dict = {
                        "epoch": epoch,
                        "global_step": epoch,
                        "lr": modeldict.optimizer.param_groups[0]["lr"],
                        "time_elapsed": timer.elapsed(),
                    }

                    if_improved = best_loss.update(
                        {
                            "train_loss": np.mean(train_record.data_dict["loss_ene"]),
                            "eval_loss": np.mean(eval_record.data_dict["loss_ene"]),
                            "tot_loss": np.mean(
                                np.concatenate(
                                    [
                                        train_record.data_dict["loss_ene"],
                                        eval_record.data_dict["loss_ene"],
                                    ]
                                )
                            ),
                        }
                    )
                    epoch_lr = experiment_dict["lr"]

                    experiment_dict.update(
                        {
                            f"train_{key}": np.mean(train_record.data_dict[key])
                            for key in train_record.data_dict.keys()
                            if key.startswith("loss_")
                        }
                    )
                    experiment_dict.update(
                        {
                            f"eval_{key}": np.mean(eval_record.data_dict[key])
                            for key in eval_record.data_dict.keys()
                            if key.startswith("loss_")
                        }
                    )
                    run.log(experiment_dict)

                    if if_improved or (epoch % (args.eval_step * 32) == 0):
                        modeldict.save_model(epoch)

                        train_record.save(
                            modeldict.dir_checkpoint / "loss" / f"train-{epoch}.csv"
                        )
                        eval_record.save(
                            modeldict.dir_checkpoint / "loss" / f"eval-{epoch}.csv"
                        )

                    print(
                        f"Epoch: {epoch:>9} "
                        f"Loss: {np.mean(train_record.data_dict["loss_ene"]):>9.2e} "
                        f"Eval: {np.mean(eval_record.data_dict["loss_ene"]):>9.2e} "
                        f"lr: {epoch_lr:>9.2e} "
                        f"{timer.measure()}",
                        flush=True,
                    )
        modeldict.scheduler.step()

        if modeldict.args.distributed:
            dist.barrier()
    torch.distributed.destroy_process_group()
