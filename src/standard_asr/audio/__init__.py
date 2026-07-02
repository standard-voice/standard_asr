# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""The audio input pipeline: from any user input to engine-ready audio.

Input types and probing, format negotiation against engine properties,
decoding and conversion, resampling, WAV encoding, and the byte-pinned PCM
wire codec.

This is an internal grouping package: import public names from
:mod:`standard_asr` (application surface) or :mod:`standard_asr.engine`
(engine-author surface) instead of from here.
"""
