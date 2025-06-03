"""Module providing a training method."""

import os
import random
import time
import numpy as np
import torch
from torchinfo import summary

import wandb

from cc2cc.utils import DataRecord
from cc2cc.utils import CUBE_SIZE
from cc2cc.utils import DataBase, ModelClass, DataBase_c, DataBase_7

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

    modeldict = ModelClass(args)

    if modeldict.model_type == "center_4":
        input_size = (302 * 75 * 10, 4)
        database_eval = DataBase_c(eval_str_dict, args)
        database_train = DataBase_c(train_str_dict, args)
    elif modeldict.model_type == "cube":
        input_size = (302 * 75 * 10, 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        database_eval = DataBase(eval_str_dict, args)
        database_train = DataBase(train_str_dict, args)
    elif modeldict.model_type == "cube_7":
        database_eval = DataBase_7(eval_str_dict, args)
        database_train = DataBase_7(train_str_dict, args)
        input_size = (302 * 75 * 10, 4, 7)
    else:
        raise ValueError(f"Unknown model type: {modeldict.model_type}")

    modeldict.init_database(database_train, database_eval)

    print(
        summary(
            modeldict.model,
            input_size=input_size,
            depth=10,
            dtypes=(
                [torch.float32] if args.precision == "float32" else [torch.float64]
            ),
            mode="train",
        )
    )

    experiment_dict = {
        "model": args.model,
        "batch_size": args.batch_size,
        "n_train": len(database_train.name_list),
        "n_eval": len(database_eval.name_list),
        "precision": args.precision,
        "basis": args.basis,
        "with_eval": args.with_eval,
        "weight_decay": args.weight_decay,
        "load": args.load,
        "jobid": os.environ.get("SLURM_JOB_ID"),
        "pid": os.getpid(),
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
            if not modeldict.with_eval:
                modeldict.scheduler.step()

            if epoch % args.eval_step == 0:
                eval_data_record_l = modeldict.eval_model()
                if modeldict.with_eval:
                    modeldict.scheduler.step(
                        np.mean([data["loss_tot"] for data in eval_data_record_l])
                    )

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
                    for key in data.keys()
                    if key.startswith("loss_")
                }
            )
            experiment_dict.update(
                {
                    f"eval_{key}": np.mean([data[key] for data in eval_data_record_l])
                    for key in data.keys()
                    if key.startswith("loss_")
                }
            )
            run.log(experiment_dict)

            time_end = time.time()
            time_elapsed = time_end - time_start
            print(
                f"Epoch: {epoch}, "
                f"Loss: {np.mean([data["loss_ene"] for data in train_data_record_l]):>5.2f}, "
                f"Eval: {np.mean([data["loss_ene"] for data in eval_data_record_l]):>5.2f}, "
                f"lr: {experiment_dict["lr"]:>5.2e}, "
                f"Speed: {time_elapsed / (epoch + 1):>5.2f}s/epoch",
                flush=True,
            )
