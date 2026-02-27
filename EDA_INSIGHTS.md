# EDA Analysis & Refined Strategy

## Critical Findings from Real Data

### 1. Position Patterns (MORE COMPLEX than assumed)

**Previous Assumption:**
- Pos 0-2: 95% letters
- Pos 3-6: 90% digits

**Reality from EDA:**
```
Position 0: ~98% letters (A, B dominant - 4000+ each)
Position 1: ~85% letters (more distributed across A-Z)
Position 2: ~70% letters (MIXED - many digits appear!)
Position 3: 100% DIGITS (uniform ~1500-1750 per digit)
Position 4: ~60% letters, ~40% digits ← CRITICAL MIXED ZONE
Position 5: ~30% letters, ~70% digits
Position 6: 100% DIGITS (uniform ~1500-1750 per digit)
```

**Brazilian Plate Format Reality:** ABC1D23 → But position 4 is AMBIGUOUS (can be letter or digit)!

---

### 2. Character Frequency Imbalance (SEVERE)

**High Frequency:**
- Letter 'A': 7500 occurrences (most common)
- Digits 0-9: 5000-7500 each (balanced)
- Letters B, C, E: 3000-3900 each

**Low Frequency:**
- Letters K, L, M, N, O: 1000-1500 each (5-7x less than 'A'!)
- Letters P, Q, R, S: 2000-2700

**Problem:** Model will overfit to A, B, C, digits → poor accuracy on rare letters K, L, M, N, O

**Impact on 90% target:** If rare letters have 60% accuracy while common ones have 95%, overall accuracy drops significantly!

---

### 3. Size Distribution Confirmed

- LR images: 32×128 (tight cluster)
- HR images: 60-160 width (spread)
- ✅ 27M param model appropriate for LR, but need strong feature extraction

---

## Refined Expert Strategy

### ❌ Previous Design (Flawed)
```
E0: Letters (pos 0-2) + Clear
E1: Letters + Degraded
E2: Digits (pos 3-6) + Clear
E3: Digits + Degraded
```

**Problems:**
1. Position 2,4,5 are MIXED → experts confused
2. Character frequency imbalance not addressed
3. Position 3,6 are pure digits but treated same as mixed positions

---

### ✅ New Design Option 1: Position-Refined + Quality

**Strategy:** Specialize by ACTUAL positional behavior + image quality

```python
E0: Start positions (0-1) + Clear    → High-confidence letter specialist
E1: Start positions (0-1) + Degraded → Robust letter specialist
E2: Pure-digit zones (3,6) + All     → Digit-only specialist (easier task)
E3: Ambiguous zones (2,4,5) + All    → Mixed letter/digit specialist (hard task)
```

**Routing Logic:**
```
Token at pos 0 (letter 'A', clear image):
  → E0 (30%) + E2 (70%) ... NO! E2 is for digits
  → E0 (80%) + E3 (20%) ... Better, but E3 is for ambiguous positions

Token at pos 4 (letter 'D' or digit '3', degraded):
  → E3 (70%) + E1 (30%) ... E3 handles ambiguity, E1 provides degraded-image skill
```

**Config:**
```python
MOE_NUM_EXPERTS = 4
MOE_TOP_K = 2
# Manually bias routing? No - let it learn from data
```

---

### ✅ New Design Option 2: Frequency-Aware + Quality

**Strategy:** Address character imbalance directly

```python
E0: Frequent chars (A,B,C,E,0-9) + Clear    → Common character fast path
E1: Frequent chars + Degraded               → Robust common character path
E2: Rare chars (K,L,M,N,O) + Clear          → Extra capacity for underrepresented chars
E3: Rare chars + Degraded                   → Hard case: rare char + bad image
```

**Advantages:**
- Directly addresses 7.5x frequency imbalance
- E2, E3 get more gradient signal for rare characters
- Load balancing naturally enforces equal attention to rare chars

**Implementation:**
Need to modify routing to consider character identity (not just position) → Too complex!

---

### ✅ New Design Option 3: Simpler - Just Fix Position Grouping

**Strategy:** Keep hybrid quality approach, but fix position assumptions

```python
E0: Letter-dominant (pos 0,1) + Clear
E1: Letter-dominant (pos 0,1) + Degraded
E2: Digit-dominant (pos 3,5,6) + Clear
E3: Digit-dominant (pos 3,5,6) + Degraded
# Let pos 2,4 (ambiguous) be routed by both position groups
```

**This is MINIMAL change** - just update position embeddings to bias routing correctly.

---

## Additional Tricks to Reach 90%

### Trick 1: Character Frequency Balancing ⭐⭐⭐

**Problem:** CTC loss treats all characters equally → model lazy, only learn A, B, 0-9

**Solution:** Weight CTC loss by inverse frequency

```python
# In trainer.py
char_counts = {
    'A': 7500, 'B': 3500, 'C': 3200, 'D': 3300, 'E': 3900,
    'K': 1000, 'L': 1100, 'M': 1200, 'N': 1100, 'O': 1400,
    '0': 5000, '1': 5600, '2': 5700, ..., '9': 5400
}

# Compute inverse weights
total = sum(char_counts.values())
char_weights = {char: total / (count * len(char_counts)) 
                for char, count in char_counts.items()}

# In CTC loss computation
# ctc_loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
# Weighted version:
loss = weighted_ctc_loss(log_probs, targets, input_lengths, target_lengths, char_weights)
```

**Expected Gain:** +2-4% (rare character accuracy improves significantly)

---

### Trick 2: Position-Constrained Beam Search ⭐⭐

**Current:** `decode_with_confidence()` only fixes confusions AFTER decoding

**Better:** Constrain decoding beam to only allow valid characters per position

```python
def decode_position_aware_beam(log_probs, beam_width=5):
    # Position 0: Only allow letters
    # Position 1: Only allow letters
    # Position 2: Allow letters + digits (most probable wins)
    # Position 3: Only allow digits
    # Position 4: Allow letters + digits
    # Position 5: Prefer digits (80% prob threshold)
    # Position 6: Only allow digits
    
    # Mask invalid characters in beam search
    for pos in range(7):
        if pos in [3, 6]:  # Pure digit positions
            log_probs[:, pos, letter_indices] = -1e9  # Block letters
        elif pos in [0, 1]:  # Strong letter positions
            log_probs[:, pos, digit_indices] = -1e9   # Block digits
```

**Expected Gain:** +1-2% (eliminates impossible sequences)

---

### Trick 3: Synthetic Data for Rare Characters ⭐⭐⭐

**Problem:** 'K' appears 1000 times, 'A' appears 7500 times → model sees 'A' during training 7.5x more

**Solution:** Oversample tracks containing rare characters

```python
# In dataset.py
def _balance_character_distribution(self, samples):
    """Oversample rare character tracks to balance frequency."""
    rare_chars = ['K', 'L', 'M', 'N', 'O', 'Q', 'U', 'V', 'W']
    rare_samples = [s for s in samples if any(c in s['text'] for c in rare_chars)]
    
    # Duplicate rare samples 3x
    balanced_samples = samples + rare_samples * 3
    random.shuffle(balanced_samples)
    return balanced_samples
```

**Expected Gain:** +3-5% (rare character accuracy 60% → 85%)

---

### Trick 4: Focal Loss for CTC ⭐⭐

**Problem:** Model overfits to easy examples (common chars, clear images)

**Solution:** Reduce loss weight on confident predictions

```python
def focal_ctc_loss(log_probs, targets, input_lengths, target_lengths, gamma=2.0):
    """Focus on hard examples."""
    ctc_loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
    
    # Compute confidence (probability of correct path)
    probs = log_probs.exp()
    pt = ... # Compute probability of ground truth sequence
    
    # Focal weighting: (1 - pt)^gamma
    focal_weight = (1 - pt) ** gamma
    
    return (ctc_loss * focal_weight).mean()
```

**Expected Gain:** +1-2% (hard examples get more attention)

---

### Trick 5: SpecAugment for LR Images ⭐⭐

**Problem:** LR images are 32×128 → very small, prone to overfitting

**Solution:** Frequency masking (horizontal bands) + time masking (vertical bands)

```python
# In transforms.py
def freq_mask(img, num_masks=1, mask_height=3):
    """Mask horizontal bands (simulate damaged plate rows)."""
    h, w = img.shape[:2]
    for _ in range(num_masks):
        y = random.randint(0, h - mask_height)
        img[y:y+mask_height, :] = 0
    return img

def time_mask(img, num_masks=1, mask_width=5):
    """Mask vertical bands (simulate occlusion)."""
    h, w = img.shape[:2]
    for _ in range(num_masks):
        x = random.randint(0, w - mask_width)
        img[:, x:x+mask_width] = 0
    return img

# In training transform
transforms.append(RandomApply([freq_mask, time_mask], p=0.3))
```

**Expected Gain:** +1-2% (better generalization to degraded images)

---

### Trick 6: Multi-Scale Training ⭐

**Problem:** LR images fixed at 32×128 → model sensitive to exact size

**Solution:** Train on multiple resolutions

```python
# In dataset.py
def get_transform(mode='train'):
    if mode == 'train':
        # Random scale between 0.9-1.1
        target_heights = [28, 30, 32, 34, 36]
        h = random.choice(target_heights)
        w = int(h * 4)  # Keep 1:4 aspect ratio
        return Resize((h, w))
    else:
        return Resize((32, 128))  # Fixed for val/test
```

**Expected Gain:** +0.5-1% (more robust to size variation)

---

## Recommended Implementation Plan

### Phase 1: Character Balancing (HIGHEST IMPACT)

1. **Implement weighted CTC loss** (Trick 1)
2. **Oversample rare character tracks** (Trick 3)

**Expected:** 78% → 83-85%

---

### Phase 2: Decoding Improvements

3. **Position-aware beam search** (Trick 2)
4. **Keep existing confusion correction** (already done)

**Expected:** 83-85% → 86-88%

---

### Phase 3: Augmentation

5. **Add SpecAugment** (Trick 5)
6. **Multi-scale training** (Trick 6)

**Expected:** 86-88% → 88-90%

---

### Phase 4: Loss Function (If still < 90%)

7. **Focal CTC loss** (Trick 4)

**Expected:** 88-90% → 90-92%

---

## Revised Expert Config

Based on EDA, the **simplest robust approach** is:

```python
# configs/config.py
MOE_NUM_EXPERTS: int = 4
MOE_TOP_K: int = 2

# Position bias for routing (not strict constraint)
# E0: Bias toward pos 0,1 (strong letters)
# E1: Bias toward pos 3,6 (pure digits)
# E2: Bias toward pos 2,4,5 (ambiguous)
# E3: Quality-aware (degraded images all positions)
```

**But ALSO add character balancing** to address frequency imbalance!

---

## Summary: What to Implement

### Immediate (High Impact):
1. ✅ **Weighted CTC Loss** → +2-4%
2. ✅ **Oversample Rare Characters** → +3-5%
3. ✅ **Position-Aware Beam Search** → +1-2%

**Total Expected:** 78% + 6-11% = **84-89%** ✅

### If Needed (Reach 90%):
4. **SpecAugment** → +1-2%
5. **Focal Loss** → +1-2%

**Total:** **87-93%** ✅✅

---

## Implementation Priority

**Start with Tricks 1+3 (character balancing)** - these directly address the 7.5x frequency imbalance revealed by EDA. This is MORE important than expert design refinement!

**Then add Trick 2 (position-aware beam search)** - leverages the position patterns from EDA.

**MoE expert design is secondary** - current config with top_k=2 is OK, but gains come from class balancing, not expert specialization.
