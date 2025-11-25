# Integration Plan: Custom Data Loader for GATraj Pipeline

This plan identifies the exact files, functions, and line numbers where you need to integrate your custom `preprocess_gat_raj_data()` + `Data_Loader` into the existing GATraj pipeline.

---

## Goal Description

Replace the original GATraj data-loading pipeline (`DataLoader_bytrajec2` from `utils.py`) with your custom loader from `dataloader.py` so the full pipeline uses:

```
CSV → preprocess_gat_raj_data() → Data_Loader → Processor → Model (with TCN) → Decoder → Loss → Train/Test
```

This integration requires changes to argument definitions in `train.py` and data loader initialization/usage in `Processor.py`.

---

## User Review Required

> [!IMPORTANT]
> **Missing Argument**: Your `Data_Loader` expects `args.total_len` (line 91 in [dataloader.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/dataloader.py#L91)), but this argument is not currently defined in [train.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/train.py). You'll need to add it or compute it as `obs_len + pred_len`.

> [!WARNING]
> **Data Format Mismatch**: The current `DataLoader_bytrajec2.get_train_batch()` returns 5 values in a specific order (see line 440-442 in [utils.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/utils.py#L440-L442)), but your custom `Data_Loader.get_train_batch()` returns 7 values (see line 198 in [dataloader.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/dataloader.py#L198)). You'll need to handle this mapping in `Processor.py`.

> [!IMPORTANT]
> **CSV Data Path**: Where is your CSV data stored? You need to specify the path to load it using `preprocess_gat_raj_data()`. The current loader uses ETH/UCY datasets from `./data/` directories (see line 21-24 in [utils.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/utils.py#L21-L24)).

---

## Proposed Changes

### Step 1: Update Arguments in train.py

**File**: [train.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/train.py)

#### Location 1.1: Add `total_len` argument

**Lines**: After line 114 (after `pred_length` definition)

**Action**: Add new argument

**Current code** (line 110-114):
```python
parser.add_argument(
    '--seq_length',default=20,type=int)
parser.add_argument(
    '--obs_length',default=8,type=int)
parser.add_argument(
    '--pred_length',default=12,type=int)
```

**Add after line 114**:
```python
parser.add_argument(
    '--total_len',default=20,type=int,
    help='Total sequence length (obs_len + pred_len)')
```

**Note**: Alternatively, you can modify your `Data_Loader.__init__` to compute `total_len` automatically as `args.obs_len + args.pred_len`.

---

#### Location 1.2: Rename/verify neighbor threshold argument

**Lines**: Line 131

**Action**: Verify the argument name matches what your loader expects

**Current code** (line 131):
```python
parser.add_argument(
    '--neighbor_thred',default=10,type=int)
```

**Issue**: Your `Data_Loader` expects `args.neighbour_threshold` (line 92 in dataloader.py) but the current arg is `neighbor_thred`. 

**Options**:
1. Update your `Data_Loader` to use `args.neighbor_thred`, OR
2. Add an alias/rename in train.py

**Recommended**: Modify your dataloader to use `args.neighbor_thred` for consistency.

---

#### Location 1.3: Add CSV data path argument

**Lines**: After line 99 (after `dataset` argument)

**Action**: Add new argument for CSV file path

**Add after line 99**:
```python
parser.add_argument(
    '--csv_data_path',default='final_surajpur_proper_reduced_2000.csv',type=str,
    help='Path to CSV data file for custom data loader')
```

---

### Step 2: Modify Processor.__init__ to Use Custom Data Loader

**File**: [Processor.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/Processor.py)

#### Location 2.1: Import custom loader and preprocessing function

**Lines**: After line 5 (after `from utils import *`)

**Action**: Add import statement

**Current code** (lines 1-10):
```python
'''
Author: Mengmeng Liu
Date: 2022/09/24
'''
from utils import *
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
```

**Add after line 5**:
```python
from dataloader import preprocess_gat_raj_data, Data_Loader
import pandas as pd
```

---

#### Location 2.2: Replace DataLoader initialization

**Lines**: Lines 12-16 (in `Processor.__init__`)

**Action**: Replace old loader with custom loader

**Current code** (lines 12-16):
```python
def __init__(self, args):
    self.args = args
    Dataloader = DataLoader_bytrajec2
    self.lr=self.args.learning_rate
    self.dataloader_gt = Dataloader(args,is_gt=True)
```

**Replace lines 14-16 with**:
```python
def __init__(self, args):
    self.args = args
    
    # Load CSV data
    csv_path = os.path.join(self.args.base_dir, self.args.csv_data_path)
    df_raw = pd.read_csv(csv_path)
    
    # Preprocess data into segments
    # For training, use training=True; for testing, use training=False
    training_mode = (self.args.phase == 'train')
    segments = preprocess_gat_raj_data(df_raw, training=training_mode)
    
    # Initialize custom data loader
    self.dataloader_gt = Data_Loader(segments, args)
    
    self.lr=self.args.learning_rate
```

**Note**: This assumes you want the same segments for train/val/test. If you need separate train/val/test splits, you'll need to split the segments or CSV data first.

---

### Step 3: Update Batch Retrieval in train_epoch

**File**: [Processor.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/Processor.py)

#### Location 3.1: Modify train_epoch batch loop

**Lines**: Lines 116-119 (in `train_epoch` function)

**Action**: Update batch retrieval and unpacking

**Current code** (lines 116-119):
```python
for batch in range(self.dataloader_gt.trainbatchnums):
    start = time.time()
    inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_train_batch(batch,epoch)
    inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
```

**Replace with**:
```python
# Note: Custom loader doesn't have trainbatchnums; calculate it
num_batches = len(self.dataloader_gt) // self.args.batch_size
for batch in range(num_batches):
    start = time.time()
    
    # Custom loader returns 7 values instead of 5
    batch_abs_gt, batch_norm_gt, nei_lists, nei_num, seq_list_gt, shift_value_gt, batch_split = self.dataloader_gt.get_train_batch()
    
    # No need to convert to tensors - already tensors from custom loader
```

**Note**: Your custom `get_train_batch()` doesn't take batch index or epoch arguments. It randomly samples internally.

---

#### Location 3.2: Update variable unpacking

**Lines**: Lines 120-121

**Action**: Remove unnecessary unpacking since we already have all variables

**Current code** (lines 120-121):
```python
if self.args.using_cuda:
    inputs_gt = tuple([i.cuda() for i in inputs_gt])
batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
```

**Replace with**:
```python
if self.args.using_cuda:
    batch_abs_gt = batch_abs_gt.cuda()
    batch_norm_gt = batch_norm_gt.cuda()
    shift_value_gt = shift_value_gt.cuda()
    seq_list_gt = seq_list_gt.cuda()
    nei_num = nei_num.cuda()
    # nei_lists is a list of tensors; move each to CUDA
    nei_lists = [nei.cuda() for nei in nei_lists]
```

---

#### Location 3.3: Update final loss calculation

**Lines**: Line 137

**Action**: Update batch count reference

**Current code** (line 137):
```python
train_loss_epoch = loss_epoch / self.dataloader_gt.trainbatchnums
```

**Replace with**:
```python
train_loss_epoch = loss_epoch / num_batches
```

---

### Step 4: Update Batch Retrieval in val_epoch

**File**: [Processor.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/Processor.py)

#### Location 4.1: Modify val_epoch batch loop

**Lines**: Lines 147-150 (in `val_epoch` function)

**Action**: Replace batch iteration with generator

**Current code** (lines 147-150):
```python
for batch in range(self.dataloader_gt.valbatchnums):
    if batch%100 == 0:
        print('testing batch',batch,self.dataloader_gt.valbatchnums)
    inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_val_batch(batch,epoch)
```

**Replace with**:
```python
batch_count = 0
for batch_data in self.dataloader_gt.get_val_batch():
    if batch_count % 100 == 0:
        num_val_batches = len(self.dataloader_gt) // self.args.batch_size
        print('validating batch', batch_count, '/', num_val_batches)
    
    batch_abs_gt, batch_norm_gt, nei_lists, nei_num, seq_list_gt, shift_value_gt, batch_split = batch_data
```

**Note**: Your `get_val_batch()` is a generator that yields batches, not an indexed function.

---

#### Location 4.2: Update variable handling and increment counter

**Lines**: Lines 151-154

**Action**: Update CUDA transfer and add batch counter

**Current code** (lines 151-154):
```python
inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
if self.args.using_cuda:
    inputs_gt=tuple([i.cuda() for i in inputs_gt])
batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
```

**Replace with**:
```python
if self.args.using_cuda:
    batch_abs_gt = batch_abs_gt.cuda()
    batch_norm_gt = batch_norm_gt.cuda()
    shift_value_gt = shift_value_gt.cuda()
    seq_list_gt = seq_list_gt.cuda()
    nei_num = nei_num.cuda()
    nei_lists = [nei.cuda() for nei in nei_lists]

batch_count += 1
```

---

### Step 5: Update Batch Retrieval in test_epoch

**File**: [Processor.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/Processor.py)

#### Location 5.1: Modify test_epoch batch loop

**Lines**: Lines 183-186 (in `test_epoch` function)

**Action**: Replace batch iteration with generator (same pattern as val_epoch)

**Current code** (lines 183-186):
```python
for batch in range(self.dataloader_gt.testbatchnums):
    if batch%100 == 0:
        print('testing batch',batch,self.dataloader_gt.testbatchnums)
    inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_test_batch(batch,epoch)
```

**Replace with**:
```python
batch_count = 0
for batch_data in self.dataloader_gt.get_test_batch():
    if batch_count % 100 == 0:
        num_test_batches = len(self.dataloader_gt) // self.args.batch_size
        print('testing batch', batch_count, '/', num_test_batches)
    
    batch_abs_gt, batch_norm_gt, nei_lists, nei_num, seq_list_gt, shift_value_gt, batch_split = batch_data
```

---

#### Location 5.2: Update variable handling and increment counter

**Lines**: Lines 187-190

**Action**: Update CUDA transfer and add batch counter (same as val_epoch)

**Current code** (lines 187-190):
```python
inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
if self.args.using_cuda:
    inputs_gt = tuple([i.cuda() for i in inputs_gt])
batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
```

**Replace with**:
```python
if self.args.using_cuda:
    batch_abs_gt = batch_abs_gt.cuda()
    batch_norm_gt = batch_norm_gt.cuda()
    shift_value_gt = shift_value_gt.cuda()
    seq_list_gt = seq_list_gt.cuda()
    nei_num = nei_num.cuda()
    nei_lists = [nei.cuda() for nei in nei_lists]

batch_count += 1
```

---

### Step 6: Fix dataloader.py Compatibility Issues

**File**: [dataloader.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/dataloader.py)

#### Location 6.1: Fix argument name mismatch

**Lines**: Lines 89-92 (in `Data_Loader.__init__`)

**Action**: Update to use consistent argument names

**Current code** (lines 89-92):
```python
self.obs_len = args.obs_len
self.pred_len = args.pred_len
self.tot_len = args.total_len
self.neighbour_threshold = args.neighbour_threshold
```

**Replace with** (to match train.py):
```python
self.obs_len = args.obs_length  # train.py uses obs_length
self.pred_len = args.pred_length  # train.py uses pred_length
self.tot_len = args.seq_length  # train.py uses seq_length (=obs+pred)
self.neighbour_threshold = args.neighbor_thred  # train.py uses neighbor_thred
```

**OR** update train.py arguments to match your dataloader (see Step 1.2 above).

---

#### Location 6.2: Fix missing else clause in preprocess

**Lines**: Line 78 (in `preprocess_gat_raj_data`)

**Action**: Add else clause for non-training mode

**Current code** (lines 59-78):
```python
if training:
    theta = random.uniform(0,2*np.pi)
    c,s = np.cos(theta) , np.sin(theta)
    R = np.array([[c,-s],[s,c]])

    #applying the rotation to the both the shifted and to the displacement

    P_shifted = P_shifted @ R.T
    P_displacement = P_displacement @ R.T

    # --- Final Segment Assembly ---
    all_segments.append({
            'agent_id': agent_id,
            'start_frame': frame_id[start_idx],
            'obs_coords_shifted': P_shifted[:obs_len],      # Input for GAT
            'obs_displacement': P_displacement[:obs_len],   # Input for TCN
            'pred_displacement_gt': P_displacement[obs_len:tot_len], # Ground Truth Target
            'shift_value': P_obs                            # The (X_obs, Y_obs) used for shifting # this single value used for the shifting
        })
```

**Replace with**:
```python
if training:
    theta = random.uniform(0,2*np.pi)
    c,s = np.cos(theta) , np.sin(theta)
    R = np.array([[c,-s],[s,c]])

    #applying the rotation to the both the shifted and to the displacement
    P_shifted = P_shifted @ R.T
    P_displacement = P_displacement @ R.T

# --- Final Segment Assembly (outside if block) ---
all_segments.append({
        'agent_id': agent_id,
        'start_frame': frame_id[start_idx],
        'obs_coords_shifted': P_shifted[:obs_len],      # Input for GAT
        'obs_displacement': P_displacement[:obs_len],   # Input for TCN
        'pred_displacement_gt': P_displacement[obs_len:tot_len], # Ground Truth Target
        'shift_value': P_obs                            # The (X_obs, Y_obs) used for shifting
    })
```

**Reason**: Currently segments are only appended during training mode, which means testing data won't be added.

---

### Step 7: Optional - Add Train/Val/Test Split Logic

**File**: [Processor.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/Processor.py)

**Lines**: In `__init__` after preprocessing (after the code added in Step 2.2)

**Action**: Split segments into train/val/test sets

**Explanation**: Currently you're using the same segments for all modes. You may want to:

1. Split your CSV data into train/val/test before preprocessing, OR
2. Split segments after preprocessing based on frame IDs or agent IDs, OR
3. Create separate CSV files for train/val/test

**Example approach** (add after segment creation in `__init__`):
```python
# Split segments for train/val/test
if self.args.phase == 'train' and self.args.ifvalid:
    split_idx = int(len(segments) * (1 - self.args.val_fraction))
    train_segments = segments[:split_idx]
    val_segments = segments[split_idx:]
    
    self.dataloader_gt = Data_Loader(train_segments, args)
    self.dataloader_val = Data_Loader(val_segments, args)
elif self.args.phase == 'train':
    self.dataloader_gt = Data_Loader(segments, args)
else:  # test mode
    self.dataloader_gt = Data_Loader(segments, args)
```

**Note**: This is optional and depends on your data organization strategy.

---

## Verification Plan

### Manual Verification Steps

1. **Verify argument parsing**
   ```bash
   python train.py --phase train --test_set 0 --num_epochs 1 --batch_size 4
   ```
   - Check that the config YAML file is created in the save directory
   - Inspect the YAML to confirm `total_len`, `csv_data_path`, `obs_length`, `pred_length`, and `neighbor_thred` are present
   - **Expected location**: `./savedata/0/GATraj/config_train.yaml`

2. **Verify data loading** (dry run)
   - Create a test script `test_loader.py` in the project root:
   ```python
   import pandas as pd
   from dataloader import preprocess_gat_raj_data, Data_Loader
   import argparse
   
   # Create minimal args
   args = argparse.Namespace(
       obs_length=8,
       pred_length=12,
       seq_length=20,
       neighbor_thred=10,
       batch_size=4
   )
   
   # Load and preprocess
   df = pd.read_csv('final_surajpur_proper_reduced_2000.csv')
   segments = preprocess_gat_raj_data(df, training=True)
   
   print(f"Total segments loaded: {len(segments)}")
   print(f"First segment keys: {segments[0].keys()}")
   print(f"Sample shift_value: {segments[0]['shift_value']}")
   
   # Create loader
   loader = Data_Loader(segments, args)
   print(f"Loader length (num frames): {len(loader)}")
   
   # Get one batch
   batch_data = loader.get_train_batch()
   print(f"Batch contains {len(batch_data)} tensors/lists")
   print(f"batch_abs_gt shape: {batch_data[0].shape}")
   print(f"batch_norm_gt shape: {batch_data[1].shape}")
   print(f"Number of scenes in batch: {len(batch_data[6])}")
   ```
   
   - Run: `python test_loader.py`
   - **Expected output**: No errors, prints showing shapes like `[20, N, 2]` where N is number of agents

3. **Verify single training step**
   - Run one epoch of training:
   ```bash
   python train.py --phase train --test_set 0 --num_epochs 1 --batch_size 4 --show_step 1
   ```
   - Monitor console output for:
     - Data loader initialization messages
     - First batch shape information
     - Training loss values (should not be NaN)
     - No crashes or index errors
   
4. **Verify validation loop**
   - Run with validation enabled:
   ```bash
   python train.py --phase train --test_set 0 --num_epochs 2 --ifvalid True --val_fraction 0.1 --batch_size 4
   ```
   - Check that validation metrics are printed after each epoch
   - Verify no shape mismatches between train and val

5. **Verify test mode**
   - First complete a training run to save model weights
   - Then run test:
   ```bash
   python train.py --phase test --test_set 0 --load_model 1 --batch_size 4
   ```
   - Check that test metrics (ADE, FDE) are computed without errors

### Automated Tests

**Note**: The current codebase does not include unit tests. Creating automated tests is recommended but optional.

**Suggested test (if you choose to create)**:
- Create `tests/test_integration.py`:
  ```python
  import torch
  from dataloader import Data_Loader, preprocess_gat_raj_data
  import pandas as pd
  import argparse
  
  def test_batch_shapes():
      args = argparse.Namespace(
          obs_length=8, pred_length=12, seq_length=20,
          neighbor_thred=10, batch_size=2
      )
      df = pd.read_csv('final_surajpur_proper_reduced_2000.csv')
      segments = preprocess_gat_raj_data(df, training=False)
      loader = Data_Loader(segments, args)
      
      batch_abs_gt, batch_norm_gt, nei_lists, nei_num, seq_list_gt, shift_value_gt, batch_split = loader.get_train_batch()
      
      assert batch_abs_gt.shape[0] == 20, "Time dimension should be 20"
      assert batch_abs_gt.shape[2] == 2, "Coordinate dimension should be 2"
      assert len(nei_lists) == len(batch_split), "One adjacency matrix per scene"
      assert nei_num.shape[0] == 8, "Neighbor counts for obs_length timesteps"
      print("✅ All shape assertions passed")
  
  if __name__ == '__main__':
      test_batch_shapes()
  ```
- Run: `python tests/test_integration.py`

---

## Summary of Changes

| File | Function/Section | Line Numbers | What to Change |
|------|-----------------|--------------|----------------|
| [train.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/train.py) | `get_parser()` | After 114 | Add `--total_len` argument |
| [train.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/train.py) | `get_parser()` | After 99 | Add `--csv_data_path` argument |
| [Processor.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/Processor.py) | Import section | After 5 | Add imports for custom loader |
| [Processor.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/Processor.py) | `__init__` | 14-16 | Replace old loader with custom loader |
| [Processor.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/Processor.py) | `train_epoch` | 116-121, 137 | Update batch retrieval and unpacking |
| [Processor.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/Processor.py) | `val_epoch` | 147-154 | Update to use generator pattern |
| [Processor.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/Processor.py) | `test_epoch` | 183-190 | Update to use generator pattern |
| [dataloader.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/dataloader.py) | `Data_Loader.__init__` | 89-92 | Fix argument name mismatches |
| [dataloader.py](file:///c:/Users/gopin/Desktop/Gopinath/Tcn_Gat_Transformer/dataloader.py) | `preprocess_gat_raj_data` | 59-78 | Move segment append outside if-block |
