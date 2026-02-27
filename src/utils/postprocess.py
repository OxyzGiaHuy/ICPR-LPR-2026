"""Post-processing utilities for OCR decoding."""
from itertools import groupby
from typing import Dict, List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Confusion-aware post-processing
# ---------------------------------------------------------------------------

# Characters that are visually confusable in low-resolution LP images.
# Key = what a bad model predicts, Value = the alternative to try.
# Bidirectional: both directions are stored explicitly.
CONFUSION_PAIRS = [
    ('0', 'O'),   # zero vs letter-O
    ('1', 'I'),   # one  vs letter-I
    ('1', 'L'),   # one  vs letter-L (less common but real)
    ('2', 'Z'),   # two  vs letter-Z
    ('5', 'S'),   # five vs letter-S
    ('6', 'G'),   # six  vs letter-G
    ('8', 'B'),   # eight vs letter-B
]

# Build lookup: char → its confusable partner(s)
from collections import defaultdict as _defaultdict
_CONFUSION_MAP: Dict[str, List[str]] = _defaultdict(list)
for _a, _b in CONFUSION_PAIRS:
    _CONFUSION_MAP[_a].append(_b)
    _CONFUSION_MAP[_b].append(_a)
CONFUSION_MAP: Dict[str, List[str]] = dict(_CONFUSION_MAP)

# Brazilian plate position rules (0-indexed on the final 7-char string):
#   Old format  AAA-DDDD : pos 0,1,2 = letter,  pos 3,4,5,6 = digit
#   Mercosur    AAADCDD  : pos 0,1,2 = letter,  pos 3 = digit,
#                          pos 4 = letter,  pos 5,6 = digit
# Conservative rule: pos 0-2 prefer letter; pos 3,5,6 prefer digit;
# pos 4 is ambiguous (Mercosur letter / old digit) → leave as-is.
POS_PREFERS_LETTER = {0, 1, 2}
POS_PREFERS_DIGIT  = {3, 5, 6}


def _apply_position_constraint(char: str, pos: int) -> str:
    """
    If `char` is confusable and its current type conflicts with what
    the position expects, swap to the confusable alternative.

    Examples (pos 0 expects letter):
        '0'  → 'O'   (digit in a letter-slot)
        '1'  → 'I'
        '2'  → 'Z'
    Examples (pos 3 expects digit):
        'O'  → '0'   (letter in a digit-slot)
        'S'  → '5'
        'B'  → '8'
    """
    if char not in CONFUSION_MAP:
        return char  # not confusable, nothing to do

    alts = CONFUSION_MAP[char]  # list of alternatives

    if pos in POS_PREFERS_LETTER and char.isdigit():
        # prefer letter alternative
        letter_alts = [a for a in alts if a.isalpha()]
        return letter_alts[0] if letter_alts else char

    if pos in POS_PREFERS_DIGIT and char.isalpha():
        # prefer digit alternative
        digit_alts = [a for a in alts if a.isdigit()]
        return digit_alts[0] if digit_alts else char

    return char


def apply_confusion_correction(pred_str: str) -> str:
    """
    Apply position-aware confusion correction to a decoded LP string.
    Only acts on standard 7-character plates; passes through anything else.

    >>> apply_confusion_correction('0BC1234')  # '0' at pos-0 → 'O'
    'OBC1234'
    >>> apply_confusion_correction('ABC0234')  # '0' at pos-3 is fine
    'ABC0234'
    >>> apply_confusion_correction('ABCO234')  # 'O' at pos-3 → '0'
    'ABC0234'
    """
    if len(pred_str) != 7:
        return pred_str  # non-standard length: don't touch
    corrected = []
    for pos, char in enumerate(pred_str):
        corrected.append(_apply_position_constraint(char, pos))
    return ''.join(corrected)


def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str],
    use_confusion_correction: bool = True,
) -> List[Tuple[str, float]]:
    """CTC decode predictions with confidence scores using greedy decoding.
    
    Args:
        preds: Log-softmax predictions of shape [batch_size, time_steps, num_classes].
        idx2char: Index to character mapping.
        use_confusion_correction: If True, apply position-aware confusion correction
            (0↔O, 1↔I, 2↔Z, 5↔S, 8↔B, 6↔G) based on plate position rules.
    
    Returns:
        List of (predicted_string, confidence_score) tuples.
    """
    probs = preds.exp()
    max_probs, indices = probs.max(dim=2)
    indices_np = indices.detach().cpu().numpy()
    max_probs_np = max_probs.detach().cpu().numpy()
    
    batch_size, time_steps = indices_np.shape
    results: List[Tuple[str, float]] = []
    
    for batch_idx in range(batch_size):
        path = indices_np[batch_idx]
        probs_b = max_probs_np[batch_idx]
        
        pred_chars = []
        confidences = []
        time_idx = 0
        
        for char_idx, group in groupby(path):
            group_list = list(group)
            group_size = len(group_list)
            
            if char_idx != 0:  # Skip blank
                pred_chars.append(idx2char.get(char_idx, ''))
                group_probs = probs_b[time_idx:time_idx + group_size]
                confidences.append(float(np.max(group_probs)))
            
            time_idx += group_size
        
        pred_str = "".join(pred_chars)

        if use_confusion_correction:
            pred_str = apply_confusion_correction(pred_str)

        confidence = float(np.mean(confidences)) if confidences else 0.0
        results.append((pred_str, confidence))
    
    return results
