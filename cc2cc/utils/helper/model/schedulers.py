import torch.optim as optim


def select_scheduler(args, optimizer, steps_per_epoch, start_update, max_lrs):
    """Build the update-based schedule selected by args.scheduler.

    Required args by schedule:
        onecycle: epoch (inclusive final epoch).
        cosine: cosine_T (epochs), cosine_eta_min.
        constant: no additional settings; one-third LR for the first five epochs.
        cosine_warm: cosine_T (epochs), cosine_T_mult, cosine_eta_min.

    steps_per_epoch converts epoch durations to optimizer updates. start_update
    resumes the schedule at the completed-update count. max_lrs supplies the
    base learning rate for each optimizer parameter group.
    """
    if args.scheduler == "onecycle":
        for group, max_lr in zip(optimizer.param_groups, max_lrs):
            group["initial_lr"] = max_lr / 25.0
            group["max_lr"] = max_lr
            group["min_lr"] = max_lr / 25.0 / 1000.0
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=(args.epoch + 1) * steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=25.0,
            final_div_factor=1000.0,
            last_epoch=start_update - 1,
        )
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.cosine_T * steps_per_epoch,
            eta_min=args.cosine_eta_min,
            last_epoch=start_update - 1,
        )
    if args.scheduler == "constant":
        total_iters = 5 * steps_per_epoch
        if 0 < start_update <= total_iters:
            for group in optimizer.param_groups:
                group["lr"] /= 3.0
        return optim.lr_scheduler.ConstantLR(
            optimizer,
            total_iters=total_iters,
            last_epoch=start_update - 1,
        )
    if args.scheduler == "cosine_warm":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.cosine_T * steps_per_epoch,
            T_mult=args.cosine_T_mult,
            eta_min=args.cosine_eta_min,
        )
        if start_update:
            scheduler.step(start_update)
        return scheduler
    raise ValueError(f"Unknown scheduler {args.scheduler}")
