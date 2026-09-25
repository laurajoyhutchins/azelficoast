"""Execution-target identity for calibrated JAX research profiles.

Generic structural costing and path selection live in azelficoast.core. This module owns
only the research/runtime observation needed to bind a calibration to one JAX target.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path


def current_jax_execution_target() -> tuple[str, str]:
    """Return the current JAX backend and a hardware-bound calibration identity."""

    import jax

    cpu_model = platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break

    devices = jax.devices()
    backend = jax.default_backend()
    payload = {
        "jax_backend": backend,
        "device_kinds": sorted(str(device.device_kind) for device in devices),
        "device_count": len(devices),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return backend, "sha256:" + hashlib.sha256(encoded).hexdigest()
