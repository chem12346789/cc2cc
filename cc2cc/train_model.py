"""Module providing a training method."""

import os
import random
import numpy as np
import torch
from torch import distributed as dist

import wandb

from cc2cc.utils import DataRecord
from cc2cc.utils import ModelClass
from cc2cc.utils.timer import Timer


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
    modeldict.init_model()
    modeldict.init_train()
    modeldict.init_database(train_str_dict, eval_str_dict)

    if modeldict.args.distributed:
        dist.barrier()

    if modeldict.local_rank == 0:
        experiment_dict = {
            "model": args.model,
            "device": args.device,
            "batch_size": args.batch_size,
            "n_train": len(modeldict.database_train),
            "n_eval": len(modeldict.database_eval),
            "precision": args.precision,
            "basis": args.basis,
            "weight_decay": args.weight_decay,
            "load": args.load,
            "jobid": os.environ.get("SLURM_JOB_ID"),
            "pid": os.getpid(),
            "rho_input": args.rho_input,
            "checkpoint": modeldict.dir_checkpoint.stem,
            "loss_multiplier_abs": modeldict.loss_multiplier_abs,
            "loss_multiplier_atomic": modeldict.loss_multiplier_atomic,
            "loss_ene": (
                "L1Loss"
                if isinstance(modeldict.loss_ene, torch.nn.L1Loss)
                else "MSELoss"
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

        run = wandb.init(
            project="DFT2CC",
            resume="allow",
            name="dft2cc",
            dir="~/wandb",
            config=experiment_dict,
            allow_val_change=True,
        )
        wandb.define_metric("*", step_metric="global_step")

        timer = Timer()
        best_loss = np.inf

    if modeldict.args.distributed:
        dist.barrier()

    for epoch in range(args.epoch + 1):
        if modeldict.args.distributed:
            modeldict.database_train.sampler.set_epoch(epoch)
            modeldict.database_eval.sampler.set_epoch(epoch)
        train_data_record_l = modeldict.train_model()
        modeldict.scheduler.step()

        if modeldict.args.distributed:
            dist.barrier()
            train_data_record_l_gathered = [[] for _ in range(dist.get_world_size())]
            dist.all_gather_object(train_data_record_l_gathered, train_data_record_l)
            if modeldict.local_rank == 0:
                train_data_record_l = []
                for data in train_data_record_l_gathered:
                    train_data_record_l.extend(data)

        if epoch % args.eval_step == 0:
            eval_data_record_l = modeldict.eval_model()
            if modeldict.args.distributed:
                dist.barrier()
                eval_data_record_l_gathered = [[] for _ in range(dist.get_world_size())]
                dist.all_gather_object(eval_data_record_l_gathered, eval_data_record_l)
                if modeldict.local_rank == 0:
                    eval_data_record_l = []
                    for data in eval_data_record_l_gathered:
                        eval_data_record_l.extend(data)

        if modeldict.local_rank == 0:
            experiment_dict = {
                "epoch": epoch,
                "global_step": epoch,
                "lr": modeldict.optimizer.param_groups[0]["lr"],
            }

            train_data_name = []
            train_data_record_new_l = {}
            for data in train_data_record_l:
                if data["name"] in train_data_name:
                    train_data_record_new_l[data["name"]].update(
                        {
                            key: train_data_record_new_l[data["name"]][key]
                            + [data[key]]
                            for key in data.keys()
                            if key.startswith("loss_")
                        }
                    )
                else:
                    train_data_name.append(data["name"])
                    train_data_record_new_l[data["name"]] = {
                        key: [data[key]]
                        for key in data.keys()
                        if key.startswith("loss_")
                    }
            train_data_record_l = []
            for name in train_data_name:
                train_data_record_l.append(
                    {
                        "name": name,
                        **{
                            key: np.mean(train_data_record_new_l[name][key])
                            for key in train_data_record_new_l[name].keys()
                        },
                    }
                )

            eval_data_name = []
            eval_data_record_new_l = {}
            for data in eval_data_record_l:
                if data["name"] in eval_data_name:
                    eval_data_record_new_l[data["name"]].update(
                        {
                            key: eval_data_record_new_l[data["name"]][key] + [data[key]]
                            for key in data.keys()
                            if key.startswith("loss_")
                        }
                    )
                else:
                    eval_data_name.append(data["name"])
                    eval_data_record_new_l[data["name"]] = {
                        key: [data[key]]
                        for key in data.keys()
                        if key.startswith("loss_")
                    }
            eval_data_record_l = []
            for name in eval_data_name:
                eval_data_record_l.append(
                    {
                        "name": name,
                        **{
                            key: np.mean(eval_data_record_new_l[name][key])
                            for key in eval_data_record_new_l[name].keys()
                        },
                    }
                )

            if epoch % args.eval_step == 0:
                experiment_dict.update(
                    {
                        f"train_{key}": np.mean(
                            [data[key] for data in train_data_record_l]
                        )
                        for key in train_data_record_l[0].keys()
                        if key.startswith("loss_")
                    }
                )
                experiment_dict.update(
                    {
                        f"eval_{key}": np.mean(
                            [data[key] for data in eval_data_record_l]
                        )
                        for key in eval_data_record_l[0].keys()
                        if key.startswith("loss_")
                    }
                )
                run.log(experiment_dict)

                epoch_loss = np.mean(
                    [data["loss_ene"] for data in eval_data_record_l]
                    + [data["loss_ene"] for data in train_data_record_l]
                )
                if epoch_loss < best_loss:
                    print(f"Loss improved: {best_loss:.4f} -> {epoch_loss:.4f}!")
                    best_loss = epoch_loss
                    modeldict.save_model(epoch)

                    data_record_train = DataRecord(
                        modeldict.dir_checkpoint / "loss" / f"train-loss-{epoch}"
                    )
                    for data in train_data_record_l:
                        data_record_train.add_data(data)
                    data_record_train.save_csv()

                    data_record_eval = DataRecord(
                        modeldict.dir_checkpoint / "loss" / f"eval-loss-{epoch}"
                    )
                    for data in eval_data_record_l:
                        data_record_eval.add_data(data)
                    data_record_eval.save_csv()

                    run.log(
                        {
                            "epoch_eval": epoch,
                            "train_loss_ene_epoch_eval": np.mean(
                                [data["loss_ene"] for data in train_data_record_l]
                            ),
                            "eval_loss_ene_epoch_eval": np.mean(
                                [data["loss_ene"] for data in eval_data_record_l]
                            ),
                        }
                    )

            print(
                f"Local_rank {modeldict.local_rank:>2}, "
                f"Epoch: {epoch:>5}, "
                f"Train: {len(train_data_record_l)}, "
                f"Eval: {len(eval_data_record_l)}, "
                f"Loss: {np.mean([data["loss_ene"] for data in train_data_record_l]):>5.2f}, "
                f"Eval: {np.mean([data["loss_ene"] for data in eval_data_record_l]):>5.2f}, "
                f"lr: {experiment_dict["lr"]:>5.2e}, "
                f"{timer.measure()}",
                flush=True,
            )

        if modeldict.args.distributed:
            dist.barrier()
    torch.distributed.destroy_process_group()
