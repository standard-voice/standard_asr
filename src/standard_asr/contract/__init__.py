# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""The protocol data model: the types every party shares.

Capability declarations, runtime parameters, transcription results, engine
properties, language handling, and the exception hierarchy. These modules are
the vocabulary of the standard -- applications, engines, and the toolchain all
speak in these types.

This is an internal grouping package: import public names from
:mod:`standard_asr` (application surface) or :mod:`standard_asr.engine`
(engine-author surface) instead of from here.
"""
