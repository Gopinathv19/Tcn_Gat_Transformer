Great — here’s a clear, ordered, **non-code** checklist you can follow to fully integrate your custom `preprocess_gat_raj_data()` + `Data_Loader` into the existing GATraj pipeline. I’ll state *exactly* what to change, where to change it, what to verify after each step, and common pitfalls to watch for. Follow these steps in order.

---

# OVERVIEW (goal)

Replace the original GATraj data-loading pipeline with your own loader so the full pipeline uses:
`CSV → preprocess_gat_raj_data() → Data_Loader → Processor → Model (with your TCN) → Decoder → Loss → Train/Test`.

---

# 1) Identify the integration points (files & functions)

Find these places in the repo (they already exist in the project you showed earlier):

* `Processor.py` — the place to load data and produce batches used by training and testing. Target these functions:

  * `load_data()` (or equivalent initializer that builds the dataset)
  * training loop methods e.g. `playtrain()`, `train_batch()` (or wherever they fetch batches)
  * testing loop method e.g. `playtest()` (or wherever they iterate test batches)
* `train.py` — where `args` are constructed and saved; ensure argument names match.
* Your custom loader file (where `preprocess_gat_raj_data` and `Data_Loader` live).

You will change logic only in `Processor.py` and `train.py` (to provide expected args), leaving model code (encoder, decoder, TCN) unchanged.

---

# 2) Prepare argument mapping (train.py)

Make sure the command-line args include names your `Data_Loader` expects. Confirm and/or add these arguments:

* `obs_len` (or `obs_length`) — length of observation (8 in your code)
* `pred_len` (or `pred_length`) — prediction horizon (12 in your code)
* `total_len` (or `seq_length`) — total length = obs_len + pred_len (your Data_Loader expects `args.total_len`)
* `batch_size` — batch size used by `get_train_batch()` etc.
* `neighbour_threshold` (or match your `neighbour_threshold` name)
* any other config used by your preprocessing (e.g., randomRotate, input_mix etc.)

Action: open `train.py` and ensure `get_parser()` includes these parameter names (or add aliases that you will map later). Save config so `Processor` sees them.

Verification: run `python train.py` (config save path should be created) — no crash.

---

# 3) Add your preprocessing call into Processor.load_data()

Goal: replace the original ETH/UCY file-reading logic with a step that calls your `preprocess_gat_raj_data()`.

Where to change:

* In `Processor.py` locate the method that reads dataset files and constructs the dataset (often called `load_data()` or inside `__init__`).

What to do (conceptually, no code here):

* Instead of reading the old ETH/UCY text files, call your CSV-reading routine (or pass the DataFrame path) and call `preprocess_gat_raj_data(df_raw, training=True/False)`.
* That function returns `segments` (list of segment dicts). Use these segments to instantiate your `Data_Loader`, e.g. `self.data_loader = Data_Loader(segments, args)`.

Points to check:

* Provide the right `args` object when constructing `Data_Loader` (it uses `args.obs_len`, `args.pred_len`, `args.total_len`, `args.neighbour_threshold`, `args.batch_size`).
* If your CSV file(s) are many, ensure you concatenate or call preprocess for each and extend `segments`.

Verification:

* After running `Processor.load_data()` in a dry-run, confirm `self.data_loader` exists and `len(self.data_loader)` > 0.

---

# 4) Replace usage of the old batch API with your Data_Loader

Where to change:

* Search for places where the Processor or training loop obtains batches. Typical names: `get_train_batch()`, `train_batch()`, `data.get_batch()` or `self.data.train_batch()`.

What to do (conceptually):

* Replace calls to the original data object with calls to your loader:

  * For training: use `self.data_loader.get_train_batch()` (or call your wrapper that returns same shape)
  * For validation: iterate `self.data_loader.get_val_batch()` (generator)
  * For testing: iterate `self.data_loader.get_test_batch()` (generator)
* Where original code expected specific return values, map them so the downstream code receives identical structures. Your `get_batch()` returns a tuple that matches typical GATraj expectations:

  * `batch_abs_gt` [T, N_total, 2]
  * `batch_norm_gt` [T, N_total, 2]
  * `nei_list_batch` (list of per-scene adjacency [obs_len, n, n])
  * `nei_num_batch` [obs_len, N_total]
  * `seq_list` [T, N_total] (binary mask)
  * `shift_value_batch` [N_total, 2]
  * `batch_split` list-of-spans

Action:

* Make sure the receiving parts of the code (the model forward and the Processor training step) use the tuple fields in the correct order and variable names. If they use different names, add a short mapping layer inside Processor, not inside model code.

Verification:

* Do a single “forward-only” pass in playtrain: call your Processor to fetch one training batch and pass it into model.forward (without backprop). Confirm shapes and no exceptions.

---

# 5) Ensure tensor ordering and shapes match model expectations

Key shape conventions your model expects (from code you showed earlier):

* Many modules expect `[batch, channels, seq_len]` for conv input, or `[N, seq_len, D]` for transformers / LSTM depending on where the permutation happens.
* Your `Data_Loader.get_batch()` returns time-major `[T, N, 2]` sometimes — confirm where the code permutes dimensions to match `Conv1d(in_channels=2, ...)`. Processor should perform necessary `permute()` operations before passing into model components.
* Make sure `seq_list` mask and `batch_split` indexing are used by the same logic you had before.

Verification:

* Print shapes at the boundary: after Processor creates tensors, print shapes and sample values, then ensure the model receives shapes it expects. Do this for one batch.

---

# 6) Handle batching of multiple scenes (scene merging)

Your `get_batch()` merges multiple scenes into a single batch by concatenating agent columns and tracking `batch_split`. Ensure the downstream code uses `batch_split` to reconstruct per-scene information when computing adjacency, losses, or per-scene decoding.

Things to check:

* If code earlier expected an adjacency matrix for the whole batch rather than per-scene list, make sure you either:

  * convert `nei_list_batch` (list-of-scenes) into a single big adjacency with zeros between scenes, or
  * ensure model code uses per-scene adjacency list (preferred, since your loader returns that).

Verification:

* Run inference and check model doesn’t assume wrong adjacency indexing (index-out-of-bounds, etc.).

---

# 7) Maintain randomness & augmentation parity

Your preprocess function does random rotation and augmentation during training. Make sure:

* The `training` flag is set appropriately when you call `preprocess_gat_raj_data()` for training vs validation/test (turn off random rotation during validation/test).
* A good pattern: create `segments_train` and `segments_val/test` separately — or have your loader perform augmentation at sample time instead of at preprocess time.

Verification:

* For deterministic debugging, set the seed and check that repeated runs produce same augmented values.

---

# 8) Update Processor saving / checkpointing / config

You already save args to YAML in `train.py`. Ensure:

* New `total_len` or `total_seq_len` param is in saved config.
* If you modify argument names, update config save/load to avoid key mismatch.
* Document new args (obs_len, pred_len, total_len, neighbour_threshold) in README or config file.

Verification:

* Start one training run, and inspect the saved config YAML to confirm all required fields exist.

---

# 9) Small compatibility fixes to watch for

When integrating, you may hit small problems — check these proactively:

* **Plotting imports**: some modules import `matplotlib` even if unused at runtime; guard or install package.
* **gcd/fractions**: old code may import deprecated names — replace with `math.gcd` or use `Fraction`.
* **Device placement**: ensure Processor moves tensors to CUDA if `args.using_cuda` is True, before model.forward.
* **Dtype**: ensure tensors are float32 (not float64) to avoid slow ops or mismatches.
* **Zero-padding mask**: your code uses zero entries to mark padded agents; confirm mask creation (`seq_list`) matches existing code's expectations.

---

# 10) Testing plan (order of tests; from easiest → full run)

1. **Unit test**: call your `preprocess_gat_raj_data()` on one CSV, create one small `segments` list, instantiate `Data_Loader`, call `get_train_batch()` and check shapes.
2. **Processor dry run**: modify `Processor.load_data()` so it constructs your loader; call `Processor.load_data()` and then call `Processor`’s method that fetches a batch and pass batch to `model.forward()` once — no backward pass.
3. **Single-step training**: run the training loop for one batch with backprop but only one step — verify loss calculation and optimizer step work and parameters update.
4. **Full epoch**: run one epoch end-to-end with training and validation.
5. **Compare results**: run with LSTM and TCN (same seed and same data) and compare a few metrics (loss, prediction visuals).
6. **Stability tests**: run multiple epochs and confirm no NaNs, exploding gradients, or shape errors.

---

# 11) Logging & visualization

* Add clear logging at points where the loader hands off to Processor (shapes, number of scenes, avg agents per scene).
* Visualize a handful of predicted vs ground-truth trajectories after a few training iterations to be sure your loader’s coordinate conventions (absolute vs shifted) match model expectations.
* If predictions look shifted or incorrect, check `shift_value` application and where the model expects absolute vs relative coordinates.

---

# 12) Clean up and documentation

After integration:

* Document in README which data files the loader expects and which args to set.
* Note default values for `obs_len`, `pred_len`, `total_len`.
* Remove debug prints and temporary mocking in tests.
* Commit the changes in a feature branch so you can revert if needed.

---

# COMMON PITFALLS & QUICK REMEDIES

* **Shape mismatch in conv1d**: conv1d expects channels-first. Remedy: permute `[T, N, 2]` → `[N, 2, T]` or follow the exact same path your model expects (Processor should handle it).
* **Wrong shift sign**: if predicted positions appear offset, check whether you subtracted anchor (P_obs) in the same direction as the model expects to reconstruct absolute positions.
* **Neighbors off-by-one**: if agents are missing from adjacency, verify mask logic that removes zero-padded agents.
* **Batch_split misuse**: if decoder uses spans but your spans indices are wrong, ensure `cur_start` and `n_s` increments match ordering.

---

# FINAL CHECKLIST (do these in order)

1. Add/confirm args in `train.py` (obs_len, pred_len, total_len, neighbor threshold, batch_size).
2. In `Processor.load_data()`, call `preprocess_gat_raj_data()` and instantiate `Data_Loader`.
3. Replace old batch-get calls with `self.data_loader.get_train_batch()` / `get_val_batch()` / `get_test_batch()`.
4. Ensure shapes and dtypes match model expectations (print-check one sample).
5. Do a forward-only pass through model with a real batch (no backprop).
6. Do a one-step training run (forward + backward + optimizer step).
7. Run a full epoch, validate, and visualize outputs.
8. Tune hyperparameters and finalize.

---

If you want, I can now produce **3 focused artifacts** (pick any):

1. A short, ready-to-paste **Processor.load_data() patch** that wires your `Data_Loader` into the project (no extra code from you).
2. A **mapping table** between your `Data_Loader` outputs and GATraj expected inputs with explicit names/ordering.
3. A **dry-run checklist** with exact terminal commands and debug-print locations so you can execute the tests step-by-step.

Which (if any) do you want next?
