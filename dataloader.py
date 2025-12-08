import pandas as pd
import numpy as np
import random 
from typing import List,Dict,Any,Tuple
import torch

def preprocess_gat_raj_data(
        df_raw: pd.DataFrame,
        training: bool = True,
        global_scale: float = 10.0,
        displacement_scale: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Preprocesses trajectory data for the GAT-TCN-Transformer model.

    Args:
        df_raw: Raw dataframe with UTM coordinates.
        training: Whether to apply random rotation.
        global_scale: Scale factor to divide global coordinates (meters -> scene units).
        displacement_scale: Scale for normalizing displacement (velocity).

    Returns:
        List of dictionaries, each representing a processed trajectory segment.
    """

    # ------------------------------------------------------------------
    # 0. GLOBAL UTM NORMALIZATION (CRITICAL)
    # ------------------------------------------------------------------
    # Step A: Convert UTM to local 0-based coordinates
    df_raw['x [m]'] = df_raw['x [m]'] - df_raw['x [m]'].min()
    df_raw['y [m]'] = df_raw['y [m]'] - df_raw['y [m]'].min()

    # Step B: Scale down large meter values (recommended: /10 or /20)
    df_raw['x [m]'] = df_raw['x [m]'] / global_scale
    df_raw['y [m]'] = df_raw['y [m]'] / global_scale

    # ------------------------------------------------------------------
    # 1. RENAME COLUMNS + SORT
    # ------------------------------------------------------------------
    needed_cols = ['Time', 'Track ID', 'x [m]', 'y [m]']
    df_cleaned = df_raw[needed_cols].copy()

    df_cleaned.rename(columns={
        'Time': 'Frame ID',
        'Track ID': 'Agent ID',
        'x [m]': 'x',
        'y [m]': 'y'
    }, inplace=True)

    df_cleaned.sort_values(by=['Frame ID', 'Agent ID'], inplace=True)

    # ------------------------------------------------------------------
    # 2. GROUP BY AGENT
    # ------------------------------------------------------------------
    groups = df_cleaned.groupby('Agent ID')

    obs_len = 8
    pred_len = 12
    tot_len = obs_len + pred_len

    all_segments = []

    # ------------------------------------------------------------------
    # 3. PROCESS EACH AGENT INTO TRAJECTORY SEGMENTS
    # ------------------------------------------------------------------
    for agent_id, agent_data in groups:
        frames = agent_data['Frame ID'].to_numpy()
        coords = agent_data[['x', 'y']].to_numpy()
        n = len(coords)

        if n < obs_len + 1:
            continue

        # Slide over the agent trajectory
        for i in range(n - obs_len):
            start = i
            end = i + obs_len
            true_end = min(i + tot_len, n)

            # Extract coordinates for this segment
            seg_abs = coords[start:true_end]

            # Pad future frames if needed
            if seg_abs.shape[0] < tot_len:
                pad = np.zeros((tot_len - seg_abs.shape[0], 2))
                seg_abs = np.vstack([seg_abs, pad])

            # ----------------------------------------------------------
            # 3A. SHIFT (RELATIVE POSITION)  → Stabilizes GAT adjacency
            # ----------------------------------------------------------
            P_obs = seg_abs[obs_len - 1]                      # anchor point
            P_shifted = seg_abs - P_obs                       # relative coords

            # ----------------------------------------------------------
            # 3B. DISPLACEMENT (VELOCITY)
            # ----------------------------------------------------------
            disp = P_shifted[1:] - P_shifted[:-1:]
            disp = np.vstack([np.zeros((1, 2)), disp])         # pad first frame

            # ----------------------------------------------------------
            # 3C. NORMALIZE DISPLACEMENT (IMPORTANT FOR TCN)
            # ----------------------------------------------------------
            disp_norm = disp / displacement_scale

            # ----------------------------------------------------------
            # 3D. DATA AUGMENTATION (RANDOM ROTATION)
            # ----------------------------------------------------------
            if training:
                theta = random.uniform(0, 2 * np.pi)
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s], [s, c]])

                P_shifted = P_shifted @ R.T
                disp_norm = disp_norm @ R.T

            # ----------------------------------------------------------
            # 3E. SAVE SEGMENT
            # ----------------------------------------------------------
            all_segments.append({
                'agent_id': agent_id,
                'start_frame': frames[start],
                'coords_abs': seg_abs.copy(),                      # scaled absolute
                'obs_coords_shifted': P_shifted[:obs_len],         # [8,2]
                'obs_displacement': disp_norm[:obs_len],           # [8,2] normalized
                'pred_displacement_gt': disp_norm[obs_len:tot_len],# [12,2] normed
                'shift_value': P_obs,                              # anchor for reconstruction
                'global_scale': global_scale,
                'displacement_scale': displacement_scale
            })

    return all_segments




class Data_Loader:
  def __init__(self,segments:List[Dict[str,Any]],args : Any):
    self.segments = segments
    self.args = args
    self.obs_len = args.obs_length
    self.pred_len = args.pred_length
    self.tot_len = self.obs_len+self.pred_len
    self.neighbour_threshold = args.neighbor_thred

    #collecting the frame from the segments

    self.segments_by_frame = self.group_by_frame(segments)
    #getting all the frame ids

    self.all_frame_ids = list(self.segments_by_frame.keys())

  def group_by_frame(self,segments:List[Dict[str,Any]]) -> Dict[int,List[Dict[str,Any]]]:
    frame_dict = {}
    for seg in segments:
      frame = seg['start_frame']
      if frame not in frame_dict:
        frame_dict[frame] = []
      frame_dict[frame].append(seg)
    return frame_dict

  def __len__(self) ->int :
    return len(self.all_frame_ids)

  def build_scene(self,scene_segments):
   t = self.tot_len
   n = len(scene_segments)

   abs_s = torch.zeros((t,n,2),dtype=torch.float32)
   norm_s = torch.zeros((t,n,2),dtype=torch.float32)
   nei_lists = torch.zeros((self.obs_len,n,n),dtype=torch.float32)

   for i, seg in enumerate(scene_segments):
    P_shifted = torch.from_numpy(seg['coords_abs']).float()
    abs_s[:,i,:] = P_shifted
    last_obs = abs_s[self.obs_len - 1, i, :].clone()  
    norm_s[:, i, :] = abs_s[:, i, :] - last_obs


   for i in range (self.obs_len):
     coords = abs_s[i]
     diff = coords[:,None] - coords[None,:]
     dist = torch.norm(diff,dim=2)
     spatial_adj = (dist < self.neighbour_threshold).float()
     spatial_adj.fill_diagonal_(0)
     mask_i = (coords.sum(dim=1) != 0).float()  
     spatial_adj = spatial_adj * mask_i[:, None] * mask_i[None, :]
     nei_lists[i] = spatial_adj
   return abs_s,norm_s,nei_lists


  def get_batch(self, frame_indices):
    """
    Builds a TRUE GATraj-style batch from multiple scenes.
    Input:
        frame_indices: list of frame indexes (batch_size scenes)
    Output:
        batch_abs_gt   [T, N_total, 2]
        batch_norm_gt  [T, N_total, 2]
        nei_list_batch list: each scene → [obs_len, n_s, n_s]
        nei_num_batch  [obs_len, N_total]
        batch_split    [[start_idx, end_idx], ...]
    """

    all_abs = []
    all_norm = []
    all_nei = []
    batch_split = []
    all_shifts =[]

    cur_start = 0

    for fi in frame_indices:
        frame_id = self.all_frame_ids[fi]
        scene = self.segments_by_frame[frame_id]

        # Build the scene (abs_s, norm_s, nei_s)
        abs_s, norm_s, nei_s = self.build_scene(scene)
        n_s = abs_s.shape[1]
        
        shift_s = torch.stack(
            [torch.from_numpy(seg['shift_value']).float() for seg in scene]
            ,dim=0
        )

        # Record scene boundaries
        batch_split.append([cur_start, cur_start + n_s])
        cur_start += n_s

        # Store scene tensors
        all_abs.append(abs_s)
        all_norm.append(norm_s)
        all_nei.append(nei_s)
        all_shifts.append(shift_s)

    # Merge scenes along agent dimension
    batch_abs_gt = torch.cat(all_abs, dim=1).float()   # [T, N_total, 2]
    batch_norm_gt = torch.cat(all_norm, dim=1).float() # [T, N_total, 2]
    shift_value_batch = torch.cat(all_shifts, dim=0).float() # [N_total, 2]
    # Build neighbor counts [obs_len, N_total]
    nei_num_batch = []
    for t in range(self.obs_len):
        per_timestep_counts = [torch.sum(nei[t], dim=1) for nei in all_nei]
        nei_num_batch.append(torch.cat(per_timestep_counts, dim=0))

    nei_num_batch = torch.stack(nei_num_batch, dim=0) 
    seq_list = (batch_abs_gt.sum(dim=2) != 0).float()
    
    nei_list_batch = all_nei
    return (batch_abs_gt, batch_norm_gt, nei_list_batch, nei_num_batch, seq_list,shift_value_batch,batch_split)

  def get_train_batch(self):
    """
    Randomly sample 'batch_size' scenes and return a merged GATraj-style batch.
    """
    idxs = np.random.choice(
        len(self.all_frame_ids),
        self.args.batch_size,
        replace=False
    )

    return self.get_batch(idxs)

  def get_val_batch(self):
    """
    Generator for validation batches.
    Iterates through the dataset in chunks of 'batch_size'.
    """
    idxs = np.arange(len(self.all_frame_ids))
    for i in range(0, len(idxs), self.args.batch_size):
        yield self.get_batch(idxs[i:i+self.args.batch_size])

  def get_test_batch(self):
    """
    Generator for test batches.
    Iterates through the dataset in chunks of 'batch_size'.
    """
    idxs = np.arange(len(self.all_frame_ids))
    for i in range(0, len(idxs), self.args.batch_size):
        yield self.get_batch(idxs[i:i+self.args.batch_size])
 


