"""GPU facts and guards: name, ``nvidia-smi``, VRAM cap, and the training-queue lock."""

import json
import logging
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import torch

log = logging.getLogger(__name__)

SMI_FIELDS = (
    "name",
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "memory.used",
    "memory.total",
)
# Left free under the cap for memory outside PyTorch's allocator (cuDNN/cuBLAS handles).
CAP_MARGIN_MB = 256


def gpu_name(device: torch.device | None = None) -> str:
    """``torch.cuda.get_device_name``, or ``"cpu"`` without CUDA."""
    return torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"


def nvidia_smi() -> dict[str, str] | None:
    """Clocks (MHz), temperature (C), power (W), utilisation (%) and memory (MiB) of GPU 0."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(SMI_FIELDS)}",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return dict(zip(SMI_FIELDS, (v.strip() for v in out.split(",")), strict=True))


def cap_to_free_vram(device: torch.device, margin_mb: int = CAP_MARGIN_MB) -> float:
    """Limit PyTorch's allocator to the VRAM free now, minus ``margin_mb``; return the cap in MB.

    The Windows driver behind WSL2 may page GPU memory out to system RAM instead of failing,
    which turns an out-of-memory into a silent many-fold slowdown. With the cap, PyTorch raises
    ``torch.OutOfMemoryError`` before that can happen.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    cap = free - margin_mb * 2**20
    torch.cuda.set_per_process_memory_fraction(cap / total, index)
    log.info(
        "VRAM: %.0f MB total, %.0f MB free; PyTorch capped at %.0f MB",
        total / 2**20,
        free / 2**20,
        cap / 2**20,
    )
    return cap / 2**20


# ----------------------------------------------------------------------------- training-queue lock

QUEUE_LOCK = ".gpu_queue.lock"  # in runs_root while run_sweep.py trains


def _pid_alive(pid: int) -> bool:
    if pid <= 0:  # os.kill(-1, ...) would address every process
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def queue_holder(runs_root: Path) -> dict | None:
    """The running training queue (``{"pid", "machine", "started"}``), or None.

    A lock left by a process that no longer runs is ignored.
    """
    path = Path(runs_root) / QUEUE_LOCK
    try:
        holder = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return holder if _pid_alive(int(holder.get("pid", -1))) else None


def require_gpu_free(runs_root: Path) -> None:
    """Refuse a second GPU job while a training queue runs (CLAUDE.md, Environment).

    Raises:
        RuntimeError: a live queue holds the lock; use ``--device cpu`` instead.
    """
    holder = queue_holder(runs_root)
    if holder:
        raise RuntimeError(
            f"the training queue is running (pid {holder['pid']}, since {holder['started']}); "
            "never start a second GPU job meanwhile, use --device cpu"
        )


@contextmanager
def queue_lock(runs_root: Path, machine: str) -> Iterator[None]:
    """Hold the training-queue lock for the duration of a sweep."""
    require_gpu_free(runs_root)
    path = Path(runs_root) / QUEUE_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {"pid": os.getpid(), "machine": machine,
            "started": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    path.write_text(json.dumps(info) + "\n")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
