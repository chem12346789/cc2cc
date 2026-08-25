"""Module providing a training method."""

import os
import random

import numpy as np
import torch
from torch import distributed as dist

import wandb

from cc2cc.utils.ModelClass import ModelClass
from cc2cc.utils.computer_info import print_computer_info
from cc2cc.utils.timer import Timer


class BestLoss:
    def __init__(self):
        self.loss_dict = {
            key: np.inf for key in ("tot_loss", "train_loss", "eval_loss")
        }

    def update(self, now_loss):
        improved = False
        for key, best in self.loss_dict.items():
            if now_loss[key] < best:
                print(f"Best {key} improved: {best:.2e} -> {now_loss[key]:.2e}")
                self.loss_dict[key] = now_loss[key]
                improved = True
        return improved


def _set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _enable_deterministic_mode():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print("Warning: Using deterministic mode, which may slow down training.")


class _Logger:
    __slots__ = ("run", "timer", "best_loss", "loss_dir", "checkpoint_stride")

    def __init__(self, run, timer, best_loss, loss_dir, checkpoint_stride):
        self.run = run
        self.timer = timer
        self.best_loss = best_loss
        self.loss_dir = loss_dir
        self.checkpoint_stride = checkpoint_stride

    @classmethod
    def setup(cls, modeldict, args):
        if modeldict.local_rank != 0 and modeldict.args.distributed:
            return None

        print_computer_info(args.device)
        experiment_dict = {
            "n_train": len(modeldict.database_train),
            "n_eval": len(modeldict.database_eval),
            "jobid": os.environ.get("SLURM_JOB_ID"),
            "pid": os.getpid(),
            "checkpoint": modeldict.dir_checkpoint.stem,
            **vars(args),
        }
        print(experiment_dict)

        run = wandb.init(
            project="DFT2CC",
            resume="allow",
            name="dft2cc",
            config=experiment_dict,
            allow_val_change=True,
        )
        wandb.define_metric("*", step_metric="global_step")
        return cls(
            run,
            Timer(),
            BestLoss(),
            modeldict.dir_checkpoint / "loss",
            args.eval_step * 32,
        )

    def log(self, modeldict, train_record, eval_record, epoch):
        stats = {
            f"{prefix}_{key}": np.mean(val)
            for prefix, record in (("train", train_record), ("eval", eval_record))
            for key, val in record.data_dict.items()
            if key.startswith("loss_")
        }
        train_loss = stats["train_loss_ene"]
        eval_loss = stats["eval_loss_ene"]
        tot_loss = np.mean(
            np.concatenate(
                [train_record.data_dict["loss_ene"], eval_record.data_dict["loss_ene"]]
            )
        )
        lr = modeldict.optimizer.param_groups[0]["lr"]
        metrics = {
            "epoch": epoch,
            "global_step": epoch,
            "lr": lr,
            "time_elapsed": self.timer.elapsed(),
            **stats,
        }
        improved = self.best_loss.update(
            {"train_loss": train_loss, "eval_loss": eval_loss, "tot_loss": tot_loss}
        )
        self.run.log(metrics)
        if improved or (self.checkpoint_stride and epoch % self.checkpoint_stride == 0):
            modeldict.save_model(epoch)
            train_record.save(self.loss_dir / f"train-{epoch}.csv")
            eval_record.save(self.loss_dir / f"eval-{epoch}.csv")
        print(
            f"Epoch: {epoch:>9} Loss: {train_loss:>9.2e} Eval: {eval_loss:>9.2e} "
            f"lr: {lr:>9.2e} {self.timer.measure()}",
            flush=True,
        )


def train_model(train_list, eval_list, args):
    """
    Train the model.
    train_list: list of training molecules
    eval_list: list of evaluation molecules
    Other parameter are from the argparse.
    """

    # 0. Init the environment
    if args.seed is not None:
        _set_seed(args.seed)
        _enable_deterministic_mode()

    modeldict = ModelClass(args)
    modeldict.init_model(init_train=True)
    modeldict.init_database(train_list, eval_list)

    is_distributed = modeldict.args.distributed
    barrier = dist.barrier if is_distributed else (lambda: None)
    barrier()

    logger = _Logger.setup(modeldict, args)
    barrier()

    for epoch in range(args.epoch + 1):
        if modeldict.args.distributed:
            modeldict.database_train.sampler.set_epoch(epoch)
            modeldict.database_eval.sampler.set_epoch(epoch)

        if epoch < modeldict.start_step:
            modeldict.scheduler.step()
            barrier()
            continue

        if_grad = epoch % args.grad_step == 0
        train_record = modeldict.train_model(if_grad=if_grad)
        barrier()

        if epoch % args.eval_step == 0 or epoch == args.epoch or epoch == 0:
            eval_record = modeldict.eval_model()
            barrier()

            if logger:
                logger.log(modeldict, train_record, eval_record, epoch)

        modeldict.scheduler.step()
        barrier()

    if is_distributed:
        dist.destroy_process_group()
