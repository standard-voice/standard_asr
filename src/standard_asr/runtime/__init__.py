# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""The request execution machinery.

The engine interface and template method (:mod:`~standard_asr.runtime.interface`),
typed engine configuration, runtime-parameter gating, the streaming session
layer, model-download policy, and credential-safe error rendering.

This is an internal grouping package: import public names from
:mod:`standard_asr` (application surface) or :mod:`standard_asr.engine`
(engine-author surface) instead of from here.
"""
