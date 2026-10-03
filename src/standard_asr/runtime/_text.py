# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""Unicode text tests the runtime shares.

The stable-text boundary check and the prompt truncation both must not cut
a string between a base character and its combining marks. This module owns
the definition of a combining mark, so the two cannot disagree about it. The
stable-text check also uses a wider test, which counts the zero width joiner
and non-joiner as well.

Both tests use the standard library ``unicodedata`` only, so their answers
follow the Unicode version of the running Python. Neither is grapheme
cluster segmentation: a cut can pass them and still fall inside one
user-perceived character. The docstring of ``validate_stable_text`` in
``standard_asr.runtime.streaming`` lists the cuts known to pass.
"""

from __future__ import annotations

import unicodedata

#: ZERO WIDTH NON-JOINER. Unicode definition D56 lets it follow a base
#: character inside a combining character sequence, as a combining mark does.
_ZWNJ = "\u200c"

#: ZERO WIDTH JOINER. It glues the characters on both of its sides into one
#: user-perceived character, as in an emoji built from several parts.
_ZWJ = "\u200d"


def is_combining_mark(ch: str) -> bool:
    """Return whether ``ch`` is a Unicode combining mark.

    The test is the general category (``Mn``, ``Mc``, or ``Me``), not
    ``unicodedata.combining``. The canonical combining class exists for
    normalization order and is ``0`` for many real marks: the Thai vowel sign
    U+0E31 and the Devanagari vowel sign U+093E both have class ``0``.

    Args:
        ch: A single character.

    Returns:
        ``True`` when ``ch`` is a nonspacing, spacing, or enclosing mark.
    """
    return unicodedata.category(ch) in ("Mn", "Mc", "Me")


def splits_combining_sequence(text: str, index: int) -> bool:
    """Return whether the boundary check rejects a cut at ``index``.

    A combining character sequence is a base character followed by combining
    marks, zero width joiners, and zero width non-joiners (Unicode definition
    D56). A cut splits one when the character after the cut is such a
    follower. A cut right after a zero width joiner is also reported,
    although it does not split a combining character sequence: the joiner
    binds the character after it into the same user-perceived character, as
    in an emoji built from several code points.

    ``False`` does not mean that the cut falls between two user-perceived
    characters. This is not grapheme cluster segmentation (Unicode Standard
    Annex #29), which needs data the standard library does not carry. Many
    cuts inside one user-perceived character pass: for example, before Thai
    SARA AM (U+0E33), inside a Devanagari conjunct after the sign U+094D,
    between the two letters of a flag, and between CR and LF. The docstring
    of ``validate_stable_text`` in ``standard_asr.runtime.streaming`` owns
    the full list.

    Args:
        text: The text being cut.
        index: The cut position, with ``0 <= index <= len(text)``. The cut
            separates ``text[:index]`` from ``text[index:]``.

    Returns:
        ``True`` when the cut separates a base character from a follower, or
        leaves a zero width joiner at the end of ``text[:index]``. ``False``
        for a cut at position ``0``.
    """
    if index <= 0:
        return False
    if text[index - 1] == _ZWJ:
        return True
    if index >= len(text):
        return False
    follower = text[index]
    return is_combining_mark(follower) or follower in (_ZWNJ, _ZWJ)
