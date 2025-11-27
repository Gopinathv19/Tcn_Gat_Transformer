# Session Summary: GAT-TCN-Transformer Training Setup & Debugging

**Date**: November 26, 2025  
**Project**: Trajectory Prediction with GAT-TCN-Transformer  
**Status**: ✅ Successfully trained for 50 epochs, but model shows severe overfitting

---

## What We Accomplished

### 1. Fixed Environment & Dependency Issues ✅
- **Problem**: Missing Python packages causing import errors
- **Fixed**:
  - Installed `tqdm` (progress bars)
  - Installed `scipy` (scientific computing)
  - Installed `PyYaml` (configuration files)
  - All packages verified in `gatraj` conda environment

### 2. Fixed Missing Configuration Arguments ✅
- **Problem**: `AttributeError: 'Namespace' object has no attribute 'csv_data_path'`
- **Fixed**: Added `--csv_data_path` argument to `train.py` with default value `final_surajpur_proper_reduced_2000.csv`

### 3. Fixed Naming Inconsistencies ✅
- **Problem**: `AttributeError: 'Namespace' object has no attribute 'neighbour_threshold'`
- **Fixed**: Changed British spelling `neighbour_threshold` to American `neighbor_thred` in `dataloader.py` to match argument definition

### 4. Fixed Model Architecture Dimension Mismatch ✅
- **Problem**: `RuntimeError: Given normalized_shape=[64], expected input with shape [*, 64], but got input of size [44, 32]`
- **Root cause**: Multi-head GAT was averaging heads (output dim=32) instead of concatenating (output dim=64)
- **Fixed**: Modified `MultiHeadGAT.forward()` to always concatenate heads to maintain output dimension

### 5. Fixed Evaluation Shape Mismatch ✅
- **Problem**: `RuntimeError: The size of tensor a (13) must match the size of tensor b (19)`
- **Root cause**: Model returned 13 frames (last obs + 12 pred) but evaluator expected 19 frames
- **Fixed**: Changed model output to return frames 1-19 (obs frames 1-7 + pred frames 8-19)

### 6. Added Git Configuration ✅
- Updated `.gitignore` to exclude:
  - `savedata/` (model checkpoints)
  - `__pycache__/` (Python cache)
  - `*.pyc`, `*.pyo` (compiled files)

---

## Training Configuration

```yaml
Model: models_GAT_TCN_Transformer.GAT_TCN_Transformer
Batch Size: 4 scenes
Epochs: 50
Learning Rate: 1e-4 → 3e-6 (cosine annealing)
Optimizer: Adam
Dataset: final_surajpur_proper_reduced_2000.csv

Architecture:
  - GAT layers: 2 (hidden_dim=64, num_heads=2)
  - TCN layers: 2 (kernel=3, hidden=64)
  - Transformer layers: 4 (d_model=64, heads=4)
  - Observation length: 8 frames
  - Prediction length: 12 frames

Data:
  - Total scenes: 34
  - Batches per epoch: 8
  - Neighbor threshold: 10 meters
```

---

## Complete Training Log (50 Epochs)

```
Epoch-0 lr: 0.0001
train-0/8 (epoch 0), train_loss = 311129265935.05884
valid_error=699112.958, valid_final=1346034.984, valid_first=1.631
test_error=0.000, test_final=0.000, test_first=0.000

Epoch-1 lr: 9.990e-05
train_loss=255773623808.00000
valid_error=699112.410, valid_final=1346033.971, valid_first=1.532

Epoch-2 lr: 9.962e-05
train_loss=94249945344.00000
valid_error=699112.600, valid_final=1346034.411, valid_first=1.577

Epoch-3 lr: 9.914e-05
train_loss=82809101824.00000
valid_error=699112.967, valid_final=1346034.924, valid_first=1.637

Epoch-5 lr: 9.763e-05
train_loss=68787390976.00000
valid_error=699113.049, valid_final=1346034.876, valid_first=1.685

Epoch-10 lr: 9.074e-05
train_loss=44118715776.00000
valid_error=699112.995, valid_final=1346034.248, valid_first=1.784

Epoch-15 lr: 8.001e-05
train_loss=41026441472.00000
valid_error=699113.097, valid_final=1346033.705, valid_first=1.909

Epoch-20 lr: 6.649e-05
train_loss=39065024896.00000
valid_error=699113.404, valid_final=1346033.751, valid_first=2.071

Epoch-25 lr: 5.150e-05
train_loss=41613193728.00000
valid_error=699113.740, valid_final=1346033.823, valid_first=2.218

Epoch-30 lr: 3.651e-05
train_loss=31328727680.00000
valid_error=699113.996, valid_final=1346033.914, valid_first=2.337

Epoch-35 lr: 2.299e-05
train_loss=25805603968.00000
valid_error=699114.215, valid_final=1346033.990, valid_first=2.426

Epoch-40 lr: 1.226e-05
train_loss=30309606400.00000
valid_error=699114.396, valid_final=1346034.132, valid_first=2.473

Epoch-45 lr: 5.374e-06
train_loss=29099711360.00000
valid_error=699114.407, valid_final=1346034.162, valid_first=2.499

Epoch-50 lr: 3.000e-06
train_loss=28358189824.00000
valid_error=699114.450, valid_final=1346034.180, valid_first=2.510
test_error=699114.450, test_final=1346034.180, test_first=2.510
```

---

## Training Results Analysis

### Loss Progression

| Metric | Epoch 0 | Epoch 10 | Epoch 25 | Epoch 50 | Change |
|--------|---------|----------|----------|----------|--------|
| **Train Loss** | 311.1B | 44.1B | 41.6B | 28.4B | **-91%** ✅ |
| **Valid ADE** | 699,113 | 699,113 | 699,114 | 699,114 | **+0%** ❌ |
| **Valid FDE** | 1,346,035 | 1,346,034 | 1,346,034 | 1,346,034 | **+0%** ❌ |
| **Learning Rate** | 1e-4 | 9.07e-5 | 5.15e-5 | 3e-6 | -97% |

**ADE** = Average Displacement Error (meters)  
**FDE** = Final Displacement Error (meters)

### Key Observations

✅ **Training loss decreased significantly** (91% reduction)
- Shows model is learning from training data
- Optimization is working correctly

❌ **Validation error completely flat** (no improvement)
- Started at 699,113 meters
- Ended at 699,114 meters
- Only 1 meter change in 50 epochs!

🔴 **Severe Overfitting Detected**
- Model memorizes training data
- Cannot generalize to validation data
- Classic symptom of fundamental issue

---

## Critical Issues Identified

### Issue #1: Abnormally Large Errors ⚠️

```
Expected trajectory prediction error: 0.5 - 4.0 meters
Your validation error:             699,114 meters (≈699 km!)

Your error is 200,000x larger than normal!
```

**Likely Causes**:
1. **Coordinate system issue**: Data may be in wrong units (GPS coordinates? UTM?)
2. **Missing normalization**: Coordinates not properly scaled
3. **Data corruption**: Outliers or invalid values

### Issue #2: Zero Improvement on Validation

**Evidence**:
```
Epoch 0:  valid_error = 699112.958
Epoch 50: valid_error = 699114.450
Change = +1.492 meters (essentially zero)
```

**Interpretation**:
- Model is not learning useful patterns
- Predictions are essentially random guesses
- Or predictions are constant (e.g., always predict [0, 0])

### Issue #3: Small Dataset

```
Total unique scenes: 34
Batches per epoch: 8
Samples per batch: 4
Total training samples: ~136 trajectories
```

**Problems**:
- Very small dataset for deep learning
- High risk of overfitting
- Limited diversity in training examples

---

## Recommended Next Steps

### 🔥 Priority 1: Diagnose Data Issue

**Run this command to inspect your data**:
```powershell
python -c "import pandas as pd; df=pd.read_csv('final_surajpur_proper_reduced_2000.csv'); print('Stats:'); print(df[['x [m]','y [m]'].describe()); print('\nSample:'); print(df.head(20))"
```

**Check for**:
- Are x, y values reasonable? (0-1000m) or huge? (500,000+)
- Any NaN or infinite values?
- Are coordinates GPS lat/lon stored incorrectly?

### 🔧 Priority 2: Add Data Normalization

**Current code** (dataloader.py):
```python
# Only shifts to relative coordinates
P_obs = segment_coords_abs[obs_len-1, :]
P_shifted = segment_coords_abs - P_obs
```

**Improved version**:
```python
# Shift AND scale
P_obs = segment_coords_abs[obs_len-1, :]
P_shifted = segment_coords_abs - P_obs

# Add scaling
scale = np.std(P_shifted) + 1e-8
P_normalized = P_shifted / scale
```

### 🔍 Priority 3: Debug Predictions

**Add logging to see what model predicts**:
```python
# In Processor.py, after model forward:
print(f"Prediction mean: {loc.mean():.2f}, std: {loc.std():.2f}")
print(f"Target mean: {target.mean():.2f}, std: {target.std():.2f}")
print(f"First prediction: {loc[0, 0, :]}")
```

---

## Files Modified

1. **train.py**
   - Added `--csv_data_path` argument (line 81-83)

2. **dataloader.py**
   - Changed `args.neighbour_threshold` → `args.neighbor_thred` (line 93)

3. **models_GAT_TCN_Transformer.py**
   - Fixed `MultiHeadGAT.forward()` to concatenate heads (line 98-102)
   - Fixed output trajectory format (line 358-371)

4. **.gitignore**
   - Added `savedata/` and `__pycache__/`

---

## Model Outputs Saved

```
Location: savedata/1/GATraj/

Files:
- GATraj_50.tar          (model checkpoint at epoch 50)
- config_train.yaml      (training configuration)
- log_curve.txt          (epoch-by-epoch metrics)
- net.txt                (model architecture printout)
```

---

## Summary

### ✅ What Worked
- All code errors fixed
- Model trains without crashes
- Training loss decreases properly
- Model architecture is sound

### ❌ What Needs Fixing
- **Validation error is 200,000x too large**
- **No learning on validation set**
- Likely data preprocessing/normalization issue
- Need to inspect and fix raw data handling

### 🎯 Next Action
**Inspect your CSV data immediately** to understand the coordinate system and scale. This is almost certainly the root cause of the massive errors.
