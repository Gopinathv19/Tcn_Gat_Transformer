# GAT-TCN-Transformer Trajectory Prediction Project Summary

## Project Overview

This is a **trajectory prediction model** that combines:
- **Graph Attention Networks (GAT)** for spatial modeling
- **Temporal Convolutional Networks (TCN)** for temporal modeling  
- **Transformer Decoder** for future trajectory prediction

**Goal**: Predict future trajectories of agents (vehicles/pedestrians) based on observed historical positions.

---

## Model Architecture

### Input
- **Observation length**: 8 frames (historical trajectory)
- **Prediction length**: 12 frames (future trajectory to predict)
- **Input features**: 2D coordinates (x, y)

### Architecture Pipeline

```
Input Trajectories [8 frames × N agents × 2D coords]
    ↓
┌─────────────────────────────────────────────┐
│  1. SPATIAL ENCODING (GAT)                  │
│  - For each timestep t in observation:     │
│    • Build spatial graph of nearby agents  │
│    • Apply 2-layer Multi-Head GAT          │
│    • Output: [N agents, 64 features]       │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  2. TEMPORAL ENCODING (TCN)                 │
│  - Stack spatial features across time      │
│  - Apply TCN (kernel=3, layers=2)          │
│  - Captures temporal patterns              │
│  - Output: [8 frames, N agents, 64 feat]   │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  3. PREDICTION (Transformer Decoder)        │
│  - Memory: TCN output (observed frames)    │
│  - Decode 12 future frames autoregressively│
│  - Output: loc [12, N, 2] + scale [12, N, 2]│
└─────────────────────────────────────────────┘
    ↓
Predicted Trajectories [12 frames × N agents × 2D coords]
```

### Key Components

#### SpatialGAT
- **Input dim**: 2 (x, y coordinates)
- **Hidden dim**: 64
- **Output dim**: 64
- **Num heads**: 2
- **Layers**: 2-layer GAT with LayerNorm

#### TemporalTCN
- **Input channels**: 64
- **Hidden channels**: 64
- **Kernel size**: 3
- **Num layers**: 2
- **Residual connections**: Yes

#### TransformerDecoder
- **d_model**: 64
- **Num heads**: 4
- **Num layers**: 4
- **Prediction length**: 12
- **Output**: Gaussian distribution (mean + variance)

### Loss Function
**Gaussian Negative Log-Likelihood (NLL) Loss**
- Models prediction uncertainty
- Outputs both mean trajectory and confidence (scale)

---

## Data Pipeline

### Raw Data Format
CSV file: `final_surajpur_proper_reduced_2000.csv`
- Columns: `Time`, `Track ID`, `x [m]`, `y [m]`
- Contains real-world trajectory data

### Data Preprocessing (`dataloader.py`)

1. **Segment Extraction**:
   - Sliding window: 8 obs + 12 pred = 20 frames per segment
   - Each segment = one agent's trajectory snippet

2. **Normalization**:
   - Anchor point: Last observed position (frame 8)
   - Shift all positions relative to anchor
   - Makes predictions translation-invariant

3. **Data Augmentation** (training only):
   - Random rotation (0 to 2π)
   - Applied to both positions and displacements

4. **Batching**:
   - Group segments by start frame (= "scene")
   - Scene = all agents starting observation at same time
   - Build spatial adjacency matrix per scene

### Spatial Graph Construction
- **Neighbor threshold**: 10 meters
- **Adjacency**: Agent j is neighbor of i if distance < 10m
- **Self-loops**: Included (agent attends to itself)
- **Per-timestep graphs**: Adjacency changes each frame

---

## Current Training Results

### Setup
- **Epochs**: 50
- **Batch size**: 4 scenes
- **Learning rate**: 1e-4 → 3e-6 (cosine annealing)
- **Optimizer**: Adam
- **Total batches per epoch**: 8

### Results Summary

| Metric | Epoch 0 | Epoch 25 | Epoch 50 |
|--------|---------|----------|----------|
| **Train Loss** | 311B | 41.6B | 28.4B |
| **Valid Error (ADE)** | 699,113 | 699,114 | 699,114 |
| **Valid Final (FDE)** | 1,346,035 | 1,346,034 | 1,346,034 |
| **Test Error (ADE)** | - | - | 699,114 |

**ADE** = Average Displacement Error (average across all predicted frames)  
**FDE** = Final Displacement Error (error at last predicted frame)

---

## Identified Issues

### 🔴 Critical Issue: Validation Error NOT Improving

**Observation**: 
- Training loss decreases significantly: 311B → 28B (90% reduction) ✅
- Validation error stays constant: ~699,114 (NO improvement) ❌
- Test error matches validation: ~699,114

**This indicates severe OVERFITTING or a fundamental problem.**

### Potential Root Causes

#### 1. **Scale/Units Problem**
- Errors are in the **hundreds of thousands**
- Data likely in **meters**
- Possible coordinate system issue (e.g., GPS coordinates stored incorrectly)
- **Evidence**: `x [m]`, `y [m]` columns suggest meter units

#### 2. **Data Normalization Issue**
```python
# Current normalization (dataloader.py line 47-48):
P_obs = segment_coords_abs[obs_len-1,:]  # anchor = last obs position
P_shifted = segment_coords_abs - P_obs   # shift to relative coords
```
- This only shifts, doesn't scale
- If coordinates are large (e.g., UTM: x=500000), shifting doesn't help
- Model may struggle with large absolute values

#### 3. **Loss Function Mismatch**
- Model outputs **Gaussian NLL loss** (probabilistic)
- But the loss values are enormous (billions)
- Validation uses **L2 distance** (different metric)
- These two metrics may be misaligned

#### 4. **Cumulative Sum Issue**
```python
# models_GAT_TCN_Transformer.py line 360:
pred_cumsum = torch.cumsum(loc, dim=0)  # cumulative sum of predictions
```
- Model predicts **per-step displacements**
- Cumulative sum assumes `loc` represents deltas
- But model may be outputting **absolute positions**
- This could cause error accumulation

#### 5. **Insufficient Training Data**
- Only **34 unique scenes** in entire dataset
- With batch_size=4: only **8 batches per epoch**
- Very small dataset → high risk of overfitting

#### 6. **Neighbor Threshold Too Large**
- Threshold = 10 meters
- May include too many irrelevant agents
- Graph becomes too dense, loses spatial structure

---

## Recommended Improvements

### 🎯 Immediate Actions

#### 1. **Data Inspection** (CRITICAL)
```python
# Add to beginning of training script:
import pandas as pd
df = pd.read_csv('final_surajpur_proper_reduced_2000.csv')
print("Data statistics:")
print(df[['x [m]', 'y [m]'].describe())
print("Sample rows:")
print(df.head(20))
```
**Check**:
- Are x, y values in reasonable range (0-100m) or huge (500000+)?
- Are there outliers or corrupted data?

#### 2. **Add Proper Normalization**
```python
# In dataloader.py preprocessing:
# Option A: Z-score normalization
mean_pos = coords.mean(axis=0)
std_pos = coords.std(axis=0)
coords_normalized = (coords - mean_pos) / (std_pos + 1e-8)

# Option B: Min-max scaling
coords_normalized = (coords - coords.min()) / (coords.max() - coords.min() + 1e-8)
```

#### 3. **Fix Loss Monitoring**
Add to `Processor.py`:
```python
# After forward pass, print intermediate values:
print(f"loc mean: {loc.mean()}, std: {loc.std()}")
print(f"target mean: {target.mean()}, std: {target.std()}")
print(f"Gaussian NLL: {loss.item()}")
```

#### 4. **Reduce Neighbor Threshold**
```python
# In train.py, change from:
--neighbor_thred 10  # too large

# To:
--neighbor_thred 2  # more selective (2 meters)
```

#### 5. **Verify Prediction Format**
Check if model outputs displacements or positions:
```python
# Add debugging in model forward():
print("First predicted displacement:", loc[0, 0, :])
print("Cumsum first step:", pred_cumsum[0, 0, :])
```

---

## Architecture Improvements (Medium Priority)

### 1. **Add Skip Connection**
```python
# In GAT_TCN_Transformer.__init__():
self.skip_linear = nn.Linear(2, hidden_size)

# In forward():
skip_features = self.skip_linear(obs_pos[-1])  # last observation
decoder_input = tcn_out + skip_features.unsqueeze(0)
```

### 2. **Increase Model Capacity**
```python
--hidden_size 128  # from 64
--tcn_layers 4     # from 2
--transformer_layers 6  # from 4
```

### 3. **Add Velocity Features**
```python
# In preprocessing:
velocity = coords[1:] - coords[:-1]  # dx, dy
features = np.concatenate([coords[1:], velocity], axis=1)  # [x, y, vx, vy]
```

### 4. **Multi-Modal Prediction**
```python
# Predict K=6 possible futures, pick best:
--num_pred 6
```

---

## Training Improvements

### 1. **Learning Rate Adjustment**
```python
# Try constant LR for debugging:
--learning_rate 1e-3  # higher initial LR
# Or use ReduceLROnPlateau instead of cosine annealing
```

### 2. **Gradient Logging**
```python
# Add to training loop:
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
```

### 3. **Data Augmentation Toggle**
```python
# Try training WITHOUT rotation augmentation:
--randomRotate False
```

---

## Expected Metrics (for comparison)

For pedestrian/vehicle trajectory prediction, typical errors:

| Dataset | ADE (meters) | FDE (meters) |
|---------|--------------|--------------|
| ETH-UCY | 0.5 - 2.0 | 1.0 - 4.0 |
| nuScenes| 0.8 - 3.0 | 1.5 - 6.0 |
| **Your model** | **699,114** 😱 | **1,346,034** 😱 |

**Your error is ~200,000x larger than expected!**

This strongly suggests a **data preprocessing or units issue**, not a model architecture problem.

---

## Next Steps for Debugging

1. ✅ Inspect raw data statistics
2. ✅ Add normalization to preprocessing
3. ✅ Verify prediction format (delta vs absolute)
4. ✅ Check loss scale is reasonable
5. ✅ Reduce neighbor threshold
6. Test with smaller learning rate
7. Visualize predictions vs ground truth

---

## Key Files Reference

- [train.py](file:///C:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/train.py) - Training script & argument parser
- [Processor.py](file:///C:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/Processor.py) - Training/validation loops
- [models_GAT_TCN_Transformer.py](file:///C:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/models_GAT_TCN_Transformer.py) - Model architecture
- [dataloader.py](file:///C:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/dataloader.py) - Data preprocessing & batching
- [utils.py](file:///C:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/utils.py) - Evaluation metrics (L2forTest)

---

## Questions for Further Investigation

1. What coordinate system is the raw data in? (GPS lat/lon? UTM? Local meters?)
2. What is the actual physical scale of the trajectories? (100m? 1km?)
3. Is the data from simulated or real-world scenarios?
4. What types of agents are being tracked? (vehicles, pedestrians, drones?)
5. Are there any known data quality issues or preprocessing steps needed?
