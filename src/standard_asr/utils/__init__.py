# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Internal utility subpackage for Standard ASR.

Houses implementation helpers shared across the runtime -- audio loading,
normalization, and WAV encoding. These are internal building blocks: the public
audio surface is re-exported from the top-level :mod:`standard_asr` package, not
imported from here.

This module exists so ``standard_asr.utils`` is a regular package with an
explicit ``__init__`` (consistent with the rest of the project), rather than an
implicit PEP 420 namespace package that some build backends and tools handle
inconsistently.
"""
