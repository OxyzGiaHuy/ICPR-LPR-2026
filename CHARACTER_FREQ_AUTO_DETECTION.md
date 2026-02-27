# Character Frequency Auto-Detection

## Tổng Quan

**Vấn đề cũ:** `CHAR_FREQUENCIES` được hardcode từ EDA → không flexible, có thể sai lệch với dataset thực tế.

**Giải pháp mới:** Tự động tính character frequencies từ training dataset + caching để nhanh.

---

## Cách Hoạt Động

### 1. Lần Đầu Chạy Training
```python
python train.py
```

**Quá trình:**
```
⚖️  CHARACTER FREQUENCY BALANCING
============================================================
📊 Computing character frequencies from 38002 samples...
   [Progress bar: 38002/38002]
   
Character distribution:
   A: 7523 (most common)
   K: 1012 (rare)
   ...
   
💾 Cached character frequencies to: data/.cache/char_frequencies.json

Character Frequency Weights
============================================================
Top 10 Highest Weights (Rare Characters):
Char   Weight   Frequency  Note
------------------------------------------------------------
K      4.892    1012       ← Rare, needs attention
N      4.431    1098       ← Rare, needs attention
...
```

**Thời gian:** ~10-15 giây để scan 38k samples lần đầu

---

### 2. Lần Sau Chạy Training (Instant!)
```python
python train.py
```

**Quá trình:**
```
⚖️  CHARACTER FREQUENCY BALANCING
============================================================
📊 Loaded character frequencies from cache: data/.cache/char_frequencies.json

Character Frequency Weights
============================================================
...
```

**Thời gian:** < 1 giây (đọc từ cache)

---

## Cache File Structure

**Location:** `data/.cache/char_frequencies.json`

**Format:**
```json
{
  "A": 7523,
  "B": 3512,
  "C": 3198,
  ...
  "0": 5089,
  "1": 5623,
  ...
}
```

---

## Khi Nào Cache Được Refresh?

Cache **tự động refresh** khi:
1. File `char_frequencies.json` không tồn tại
2. Xóa cache thủ công (để force recompute)

Cache **KHÔNG** tự động refresh khi:
- Dataset thay đổi (phải xóa cache thủ công)
- Sample index thay đổi (phải xóa cache thủ công)

---

## Xóa Cache (Force Recompute)

### Windows:
```powershell
Remove-Item data\.cache\char_frequencies.json
python train.py
```

### Linux/Mac:
```bash
rm data/.cache/char_frequencies.json
python train.py
```

---

## Fallback Behavior

Nếu **compute từ dataset fail** (ví dụ: dataset corrupt), tự động dùng hardcoded values từ EDA:

```python
# In character_balance.py
CHAR_FREQUENCIES_FALLBACK = {
    'A': 7500, 'B': 3500, ..., 'K': 1000, ...
}
```

**Log khi dùng fallback:**
```
⚠️  Failed to compute frequencies from dataset: [error message]
ℹ️  Using fallback character frequencies from EDA
```

---

## API Changes

### Old (Hardcoded):
```python
from src.utils.character_balance import CHAR_FREQUENCIES

weights = compute_character_weights(
    config.CHAR2IDX,
    frequencies=CHAR_FREQUENCIES  # Always same values
)
```

### New (Auto-Computed):
```python
from src.utils.character_balance import compute_char_frequencies_from_dataset

# Compute once from dataset
char_freqs = compute_char_frequencies_from_dataset(
    train_dataset,
    cache_path="data/.cache/char_frequencies.json"
)

# Use actual frequencies
weights = compute_character_weights(
    config.CHAR2IDX,
    frequencies=char_freqs  # Real distribution!
)
```

---

## Benefits

### ✅ Accurate
- Uses **actual** character distribution from your training data
- No assumptions or hardcoded values

### ✅ Fast
- First run: ~10-15 seconds (one-time cost)
- Subsequent runs: < 1 second (cached)

### ✅ Flexible
- Works with any dataset (not just Brazilian plates)
- Automatically adapts to data changes

### ✅ Reliable
- Falls back to EDA values if computation fails
- Prints clear messages about what's happening

---

## Example Training Output

```
[TRAIN] Loaded 19001 tracks.
  💾 Loading cached index from data/.cache/sample_index_19001tracks_train.json...
  ✅ Loaded 38002 samples from cache (instant!)
  📊 Rare character oversampling: 38002 -> 52314 samples (+14312)
-> Total: 52314 samples.

============================================================
⚖️  CHARACTER FREQUENCY BALANCING
============================================================
📊 Loaded character frequencies from cache: data/.cache/char_frequencies.json
Smoothing factor: 0.1

Character Frequency Weights
============================================================
Top 10 Highest Weights (Rare Characters):
Char   Weight   Frequency  Note
------------------------------------------------------------
K      4.892    1012       ← Rare, needs attention
N      4.438    1096       ← Rare, needs attention
L      4.438    1096       ← Rare, needs attention
M      4.068    1189       ← Rare, needs attention
...

Top 10 Lowest Weights (Common Characters):
Char   Weight   Frequency  Note
------------------------------------------------------------
A      0.651    7523       ← Common, already learned
7      0.842    5812       ← Common, already learned
...

Statistics:
  Mean weight: 1.000
  Std weight:  1.247
  Min weight:  0.651 (A)
  Max weight:  4.892 (K)
  Weight range: 7.51x
============================================================

Creating ResTranMoE model (MoE with 4 experts, top-k=2)...
...
```

---

## Deprecated Functions

### `analyze_character_distribution()` → Use `compute_char_frequencies_from_dataset()`
```python
# Old (deprecated)
from src.utils.character_balance import analyze_character_distribution
freqs = analyze_character_distribution(dataset, char2idx)

# New (recommended)
from src.utils.character_balance import compute_char_frequencies_from_dataset
freqs = compute_char_frequencies_from_dataset(dataset, cache_path="...")
```

### `CHAR_FREQUENCIES` → Use `CHAR_FREQUENCIES_FALLBACK`
```python
# Old (deprecated constant)
from src.utils.character_balance import CHAR_FREQUENCIES

# New (fallback constant)
from src.utils.character_balance import CHAR_FREQUENCIES_FALLBACK
```

---

## Implementation Details

### File: [src/utils/character_balance.py](src/utils/character_balance.py)

**New function:**
```python
def compute_char_frequencies_from_dataset(
    dataset,
    cache_path: Optional[str] = None
) -> Dict[str, int]:
    """
    Compute character frequencies from dataset (with caching).
    
    1. Try load from cache
    2. If not found, scan dataset.samples
    3. Count character occurrences
    4. Save to cache
    5. Return frequencies dict
    """
```

**Updated function:**
```python
def compute_character_weights(
    char2idx: Dict[str, int],
    frequencies: Optional[Dict[str, int]] = None,  # Now optional!
    smoothing: float = 0.1
) -> torch.Tensor:
    """
    If frequencies=None, uses CHAR_FREQUENCIES_FALLBACK
    """
```

### File: [src/training/trainer.py](src/training/trainer.py)

**Updated initialization:**
```python
if config.USE_CHARACTER_BALANCING:
    # 1. Compute/load frequencies
    char_frequencies = compute_char_frequencies_from_dataset(
        train_loader.dataset,
        cache_path="data/.cache/char_frequencies.json"
    )
    
    # 2. Compute weights from actual frequencies
    self.char_weights = compute_character_weights(
        config.CHAR2IDX,
        frequencies=char_frequencies,  # <-- Real data!
        smoothing=config.CHAR_WEIGHT_SMOOTHING
    ).to(self.device)
```

---

## Troubleshooting

### Issue 1: "Computing character frequencies from 0 samples"
**Cause:** Dataset not loaded correctly  
**Fix:** Check dataset initialization, ensure `self.samples` is populated

### Issue 2: Cache not updating after dataset change
**Cause:** Cache file still exists with old frequencies  
**Fix:** Delete `data/.cache/char_frequencies.json` manually

### Issue 3: "Failed to compute frequencies"
**Cause:** Dataset.samples structure changed or corrupt  
**Fix:** Code falls back to `CHAR_FREQUENCIES_FALLBACK` automatically

---

## Migration Guide

No code changes needed! Just run training:
```powershell
python train.py
```

First run will compute + cache frequencies. All subsequent runs use cached values.

**Optional:** If you want to force recompute (e.g., after dataset update):
```powershell
Remove-Item data\.cache\char_frequencies.json
python train.py
```

---

## Summary

**Before:** Hardcoded frequencies → inflexible  
**After:** Auto-computed + cached → accurate, fast, flexible ✅

Vẫn giữ fallback values từ EDA nếu cần, nhưng mặc định dùng real data distribution!
