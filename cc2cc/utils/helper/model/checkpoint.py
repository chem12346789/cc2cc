import random
import shutil
import time
from io import BytesIO

import numpy as np
import torch
import torch.distributed as dist

from cc2cc.utils.helper.model.state_dict import normalize_state_dict


def load_model(model, checkpoints_path):
    """Load weights and, when resuming, optimizer and RNG state."""
    args = model.args
    resume = getattr(args, "if_resume", False)
    if not args.load:
        if resume:
            raise ValueError("Resuming training requires --load and --load_epoch.")
        model.print("No checkpoint specified, starting from scratch.")
        return

    checkpoint_dir = (checkpoints_path / f"checkpoint_{args.load}").resolve()
    checkpoint_path = checkpoint_dir / f"{args.load_epoch}.pth"
    model.print(f"Checking path {checkpoint_path}")
    if not checkpoint_path.exists():
        if resume:
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        model.print("Model not found, starting from scratch.")
        return

    model.print("Loading model from path")
    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=True
    )
    model.state_dict = normalize_state_dict(checkpoint["state_dict"])
    args.model = checkpoint["model"]
    if resume:
        model.optimizer_state_dict = checkpoint.get("optimizer")
        if model.optimizer_state_dict is None:
            raise ValueError(
                f"Resume checkpoint has no optimizer state: {checkpoint_path}"
            )
        model.rng_state_dict = checkpoint.get("rng_state")
    model.print(f"Model loaded from {checkpoint_path} with model {args.model}")


class _CheckpointSizeBuffer(BytesIO):
    def __init__(self):
        super().__init__()
        self.size = 0

    def write(self, data):
        # Discard streamed tensor bytes instead of buffering the whole checkpoint.
        size = len(data)
        self.size += size
        return size


def capture_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    state = {
        "python": torch.tensor(
            (python_state[0], *python_state[1]),
            dtype=torch.int64,
            device="cpu",
        ),
        "python_gauss": torch.tensor(
            (python_state[2] is not None, python_state[2] or 0.0),
            dtype=torch.float64,
            device="cpu",
        ),
        "numpy_name": torch.tensor(
            list(numpy_state[0].encode("ascii")),
            dtype=torch.uint8,
            device="cpu",
        ),
        "numpy_keys": torch.tensor(
            numpy_state[1].tolist(),
            dtype=torch.int64,
            device="cpu",
        ),
        "numpy_meta": torch.tensor(
            (numpy_state[2], numpy_state[3]),
            dtype=torch.int64,
            device="cpu",
        ),
        "numpy_gauss": torch.tensor(
            (numpy_state[4],),
            dtype=torch.float64,
            device="cpu",
        ),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state().cpu()
    return state


def restore_rng_state(rng_state_dict):
    if rng_state_dict is None:
        return

    state = rng_state_dict
    if isinstance(state, list):
        if dist.is_initialized() and len(state) != dist.get_world_size():
            raise ValueError(
                "Checkpoint RNG state rank count differs from the current process group."
            )
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank >= len(state):
            raise ValueError(
                "Checkpoint RNG state has fewer ranks than the current process group."
            )
        state = state[rank]
    python_state = state["python"].cpu().tolist()
    python_gauss = state["python_gauss"].cpu().tolist()
    random.setstate(
        (
            python_state[0],
            tuple(python_state[1:]),
            python_gauss[1] if python_gauss[0] else None,
        )
    )
    numpy_name = bytes(state["numpy_name"].cpu().tolist()).decode("ascii")
    numpy_meta = state["numpy_meta"].cpu().tolist()
    np.random.set_state(
        (
            numpy_name,
            np.asarray(state["numpy_keys"].cpu().numpy(), dtype=np.uint32),
            numpy_meta[0],
            numpy_meta[1],
            state["numpy_gauss"].cpu().item(),
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"].cpu())


def estimate_checkpoint_size(model_name, optimizer, state_dict, rng_state):
    payload = {
        "state_dict": state_dict,
        "model": model_name,
        "optimizer": optimizer.state_dict(),
        "rng_state": rng_state,
    }
    with _CheckpointSizeBuffer() as buffer:
        torch.save(payload, buffer)
        return max(buffer.size, 1 << 20)


def save_model(model, epoch):
    if not model.dir_checkpoint.exists():
        model.print(f"Directory {model.dir_checkpoint} not found. Created!")
        (model.dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)

    state_dict = normalize_state_dict(model.model.state_dict())
    checkpoint_path = model.dir_checkpoint / f"{epoch}.pth"
    rng_state = model.rng_state_for_save or capture_rng_state()
    payload = {
        "state_dict": state_dict,
        "model": model.args.model,
        "optimizer": model.optimizer.state_dict(),
        "rng_state": rng_state,
    }
    required_bytes = estimate_checkpoint_size(
        model.args.model, model.optimizer, state_dict, rng_state
    )

    while True:
        free_bytes = shutil.disk_usage(model.dir_checkpoint).free
        if free_bytes >= required_bytes:
            try:
                torch.save(payload, checkpoint_path)
                return
            except OSError as exc:
                msg = str(exc).lower()
                if "no space left" in msg or "file write failed" in msg:
                    model.print(
                        "Disk space became insufficient while writing checkpoint; "
                        f"waiting for free disk before retrying {checkpoint_path}."
                    )
                    time.sleep(30)
                    continue
                raise

        model.print(
            "Waiting for free disk space before saving checkpoint: "
            f"{free_bytes} bytes free, {required_bytes} bytes required."
        )
        time.sleep(30 * 60)  # Wait for 30 minutes before checking again
