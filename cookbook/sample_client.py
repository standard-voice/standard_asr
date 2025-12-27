"""Minimal client showcasing Standard ASR entrypoint discovery.

Run this after installing a plugin (for example the bundled ``std-dummy-asr``):

```bash
uv run uv pip install -e cookbook/std_dummy_asr
uv run python cookbook/sample_client.py
```
"""

from __future__ import annotations

import logging

import numpy as np

from standard_asr import discover_models


def main() -> None:
    """Discover installed Standard ASR models and run a quick smoke test.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If no Standard ASR models are installed.
    """
    logging.basicConfig(level=logging.INFO)

    registry = discover_models()
    if not registry.names():
        raise SystemExit(
            "No Standard ASR models are installed. Did you install a plugin?"
        )

    target = registry.names()[0]
    print(f"Using model: {target}")

    asr = registry.create(target)
    dummy_audio = np.zeros(16_000, dtype=np.float32)
    result = asr.transcribe(dummy_audio)

    print(f"Transcript: {result.text}")


if __name__ == "__main__":  # pragma: no cover - example script
    main()
