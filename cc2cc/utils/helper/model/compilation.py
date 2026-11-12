from typing import cast

import torch


def maybe_compile_model(model):
    if model.args.if_grad:
        return

    try:
        compiled_model = torch.compile(model.model)
        model.model = cast(torch.nn.Module, compiled_model)
        model.print("torch.compile enabled; compilation occurs on first use.")
    except Exception as exc:  # pragma: no cover - fallback path
        model.print(f"torch.compile failed ({exc}). Using eager mode.")
