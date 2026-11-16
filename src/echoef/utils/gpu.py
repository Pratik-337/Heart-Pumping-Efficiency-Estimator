"""GPU facts and guards: name, ``nvidia-smi`` snapshot, cap at the VRAM that is really free."""

import logging
import subprocess

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
