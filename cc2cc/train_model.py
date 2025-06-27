"""Module providing a training method."""

import os
import random
import time
import numpy as np
import torch

import wandb

from cc2cc.utils import DataRecord
from cc2cc.utils import ModelClass


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
    modeldict.init_model(args)
    modeldict.init_train(args)
    modeldict.init_database(args, train_str_dict, eval_str_dict)

    experiment_dict = {
        "model": args.model,
        "batch_size": args.batch_size,
        "n_train": len(modeldict.database_train),
        "n_eval": len(modeldict.database_eval),
        "precision": args.precision,
        "basis": args.basis,
        "weight_decay": args.weight_decay,
        "load": args.load,
        "jobid": os.environ.get("SLURM_JOB_ID"),
        "pid": os.getpid(),
        "rho_dft": args.rho_dft,
        "checkpoint": modeldict.dir_checkpoint.stem,
        "loss_multiplier_abs": modeldict.loss_multiplier_abs,
        "loss_multiplier_atomic": modeldict.loss_multiplier_atomic,
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

    with wandb.init(
        project="DFT2CC",
        resume="allow",
        name="dft2cc",
        dir="/home/chenzihao/raid/tmp",
        config=experiment_dict,
        allow_val_change=True,
    ) as run:
        wandb.define_metric("*", step_metric="global_step")
        print(f"Start training at {modeldict.dir_checkpoint}")
        time_start = time.time()

        for epoch in range(args.epoch + 1):
            train_data_record_l = modeldict.train_model()
            modeldict.scheduler.step()

            if epoch % args.eval_step == 0:
                eval_data_record_l = modeldict.eval_model()

                if epoch % (args.eval_step * 50) == 0:
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

                    experiment_dict = {
                        "epoch_eval": epoch,
                        "train_loss_ene_epoch_eval": np.mean(
                            [data["loss_ene"] for data in train_data_record_l]
                        ),
                        "eval_loss_ene_epoch_eval": np.mean(
                            [data["loss_ene"] for data in eval_data_record_l]
                        ),
                        "lr": modeldict.optimizer.param_groups[0]["lr"],
                    }
                    run.log(experiment_dict)

            experiment_dict = {
                "epoch": epoch,
                "global_step": epoch,
                "lr": modeldict.optimizer.param_groups[0]["lr"],
            }
            experiment_dict.update(
                {
                    f"train_{key}": np.mean([data[key] for data in train_data_record_l])
                    for key in train_data_record_l[0].keys()
                    if key.startswith("loss_")
                }
            )
            experiment_dict.update(
                {
                    f"eval_{key}": np.mean([data[key] for data in eval_data_record_l])
                    for key in eval_data_record_l[0].keys()
                    if key.startswith("loss_")
                }
            )
            run.log(experiment_dict)

            time_end = time.time()
            time_elapsed = time_end - time_start
            print(
                f"Epoch: {epoch:>5}, "
                f"Loss: {np.mean([data["loss_ene"] for data in train_data_record_l]):>5.2f}, "
                f"Eval: {np.mean([data["loss_ene"] for data in eval_data_record_l]):>5.2f}, "
                f"lr: {experiment_dict["lr"]:>5.2e}, "
                f"Speed: {time_elapsed / (epoch + 1):>5.2f}s/epoch",
                flush=True,
            )
        torch.distributed.destroy_process_group()
