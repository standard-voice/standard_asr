# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Minimal client showcasing Standard ASR entrypoint discovery.

Run this after installing the bundled ``std-dummy-asr`` demo plugin:

```bash
uv run uv pip install -e cookbook/std_dummy_asr
uv run python cookbook/sample_client.py
```
"""

from __future__ import annotations

import logging

import numpy as np

from standard_asr import AudioArray, RuntimeParams, discover_models

#: The demo model this client drives. Selecting a model explicitly (rather than
#: ``registry.names()[0]``) keeps the example deterministic and is what real
#: applications do -- they request a specific ``engine_id/model_name`` key.
MODEL = "dummy/echo"


def main() -> None:
    """Discover installed Standard ASR models and run a quick smoke test.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If the demo model is not installed.
    """
    logging.basicConfig(level=logging.INFO)

    registry = discover_models()
    if MODEL not in registry.names():
        raise SystemExit(
            f"Model {MODEL!r} is not installed. Install the demo plugin with "
            "'uv run uv pip install -e cookbook/std_dummy_asr'. "
            f"Discovered models: {registry.names() or '<none>'}."
        )

    print(f"Using model: {MODEL}")
    asr = registry.create(MODEL)

    # Wrap a bare waveform as AudioArray with its sample rate. A bare ndarray has
    # no sample rate, which the strict policy (the default) rejects; AudioArray
    # (or a ``(samples, sample_rate)`` tuple) states it explicitly.
    samples = np.zeros(16_000, dtype=np.float32)
    result = asr.transcribe(AudioArray(samples, 16_000), RuntimeParams(language="en"))

    print(f"Transcript: {result.text}")


if __name__ == "__main__":  # pragma: no cover - example script
    main()
