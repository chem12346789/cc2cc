import torch.optim as optim


def select_optimizer(args, model):
    """Build args.optimizer using lr, muon_lr, and weight_decay settings."""
    if args.optimizer == "Muon":
        from pytorch_optimizer import Muon

        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            if param.ndim >= 2 and "mixing_weight" not in name.split("."):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        # Muon orthogonalizes updates in bfloat16; model weights remain float64.
        return Muon(
            [
                dict(params=adamw_params, lr=args.lr, use_muon=False),
                dict(params=muon_params, lr=args.muon_lr, use_muon=True),
            ],
            weight_decay=args.weight_decay,
            adamw_wd=args.weight_decay,
            adamw_betas=(0.9, 0.999),
            adamw_eps=1e-8,
        )

    optimizer_cls = {
        "AdamW": optim.AdamW,
        "Adafactor": getattr(optim, "Adafactor", None),
    }.get(args.optimizer)
    if optimizer_cls is None:
        raise ValueError(f"Unknown optimizer {args.optimizer}")
    return optimizer_cls(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
