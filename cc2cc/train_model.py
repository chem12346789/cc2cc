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
    __slots__ = (
        "run",
        "timer",
        "best_loss",
        "loss_dir",
        "checkpoint_stride",
        "wandb_kwargs",
    )

    def __init__(
        self, run, timer, best_loss, loss_dir, checkpoint_stride, wandb_kwargs
    ):
        self.run = run
        self.timer = timer
        self.best_loss = best_loss
        self.loss_dir = loss_dir
        self.checkpoint_stride = checkpoint_stride
        self.wandb_kwargs = dict(wandb_kwargs)

    @staticmethod
    def _finish_wandb_run(run):
        if run is None:
            return
        try:
            run.finish(quiet=True)
        except Exception:
            pass

    @classmethod
    def _open_wandb_run(cls, wandb_kwargs, *, mode=None, run_id=None):
        kwargs = dict(wandb_kwargs)
        if mode is not None:
            kwargs["mode"] = mode
        if run_id:
            kwargs["id"] = run_id
            kwargs["resume"] = "allow"
        run = wandb.init(**kwargs)
        wandb.define_metric("*", step_metric="global_step")
        return run

    @classmethod
    def _init_wandb_run(cls, wandb_kwargs):
        try:
            return cls._open_wandb_run(wandb_kwargs)
        except Exception as exc:
            print(
                f"Warning: wandb online init failed ({exc}). "
                "Falling back to offline mode.",
                flush=True,
            )
            try:
                return cls._open_wandb_run(wandb_kwargs, mode="offline")
            except Exception as offline_exc:
                print(
                    f"Warning: wandb offline init failed ({offline_exc}). "
                    "Disable wandb logging.",
                    flush=True,
                )
                return None

    def _switch_to_offline_mode(self, metrics):
        prev_run_id = getattr(self.run, "id", None) if self.run is not None else None
        self._finish_wandb_run(self.run)
        self.run = None

        last_exc = None
        retry_ids = ([prev_run_id] if prev_run_id else []) + [None]
        for run_id in retry_ids:
            try:
                self.run = self._open_wandb_run(
                    self.wandb_kwargs, mode="offline", run_id=run_id
                )
                self.run.log(metrics)
                print("Warning: switched wandb to offline mode.", flush=True)
                return
            except Exception as exc:
                last_exc = exc
                self._finish_wandb_run(self.run)
                self.run = None

        print(
            f"Warning: wandb offline switch failed ({last_exc}). "
            "Disable wandb logging for remaining epochs.",
            flush=True,
        )

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

        wandb_kwargs = {
            "project": "DFT2CC",
            "resume": "allow",
            "name": "dft2cc",
            "config": experiment_dict,
            "allow_val_change": True,
        }
        run = cls._init_wandb_run(wandb_kwargs)
        return cls(
            run,
            Timer(),
            BestLoss(),
            modeldict.dir_checkpoint / "loss",
            args.eval_step * 32,
            wandb_kwargs,
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
        if self.run is not None:
            try:
                self.run.log(metrics)
            except Exception as exc:
                print(
                    f"Warning: wandb.log failed ({exc}). "
                    "Try switching to offline mode.",
                    flush=True,
                )
                self._switch_to_offline_mode(metrics)
        if improved or (self.checkpoint_stride and epoch % self.checkpoint_stride == 0):
            modeldict.save_model(epoch)
            train_record.save(self.loss_dir / f"train-{epoch}.csv")
            eval_record.save(self.loss_dir / f"eval-{epoch}.csv")
        print(
            f"Epoch: {epoch:>9} Loss: {train_loss:>9.2e} Eval: {eval_loss:>9.2e} "
            f"lr: {lr:>9.2e} {self.timer.measure()}",
            flush=True,
        )


def train_model(train_str_dict, eval_str_dict, args):
    """
    Train the model.
    train_str_dict: list of training molecules
    eval_str_dict: list of evaluation molecules
    Other parameter are from the argparse.
    """

    # 0. Init the environment
    if args.seed is not None:
        _set_seed(args.seed)
    if args.deterministic:
        _enable_deterministic_mode()

    modeldict = ModelClass(args)
    modeldict.init_model(init_train=True)
    modeldict.init_database(train_str_dict, eval_str_dict)

    is_distributed = modeldict.args.distributed
    barrier = dist.barrier if is_distributed else (lambda: None)
    barrier()

    logger = _Logger.setup(modeldict, args)
    barrier()

    for epoch in range(args.epoch + 1):
        if modeldict.args.distributed:
            modeldict.database_train.sampler.set_epoch(epoch)
            modeldict.database_eval.sampler.set_epoch(epoch)

        if epoch <= modeldict.start_step:
            modeldict.scheduler.step()
            barrier()
            continue

        train_record = modeldict.train_model()
        barrier()

        if epoch % args.eval_step == 0:
            eval_record = modeldict.eval_model()
            barrier()

            if logger:
                logger.log(modeldict, train_record, eval_record, epoch)

        modeldict.scheduler.step()
        barrier()

    if is_distributed:
        dist.destroy_process_group()
