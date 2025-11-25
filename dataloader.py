import pandas as pd
import numpy as np
import random 
from typing import List,Dict,Any,Tuple
import torch

def preprocess_gat_raj_data(df_raw:pd.DataFrame , training:bool=True)-> list:
  pred_len = 12
  obs_len = 8
  tot_len = pred_len + obs_len

  essential_columns = ['Time','Track ID','x [m]','y [m]']
  df_cleaned = df_raw[essential_columns].copy()

  column_maping = {'Time':'Frame ID','Track ID':'Agent ID','x [m]':'x','y [m]':'y'}

  df_cleaned.rename(columns=column_maping,inplace=True)
  df_cleaned.sort_values(by=['Frame ID','Agent ID'],inplace=True)
  df_group_by_frame_id = df_cleaned.groupby('Agent ID')

  all_segments = []

  for agent_id,agent_data in df_group_by_frame_id:
    frame_id = agent_data['Frame ID'].to_numpy()
    coords = agent_data[['x','y']].to_numpy()
    num_frames = len(frame_id)

    if num_frames < obs_len+1:
      continue

    for i in range (num_frames-obs_len):
      start_idx = i
      end_idx= i+obs_len

      actual_end_indx = min(i+tot_len,num_frames)

      segment_coords_abs = coords[start_idx:actual_end_indx]
      segment_frame_ids = frame_id[start_idx:actual_end_indx]

      if segment_coords_abs.shape[0] < tot_len:
        padding_needed = tot_len-segment_coords_abs.shape[0]
        padding = np.zeros((padding_needed,2))
        segment_coords_abs=np.vstack([segment_coords_abs,padding])

      # in this step we are subtracting the anchor point or the final point with all other points in the coords ok
      # in this step we have normalized the value , from the anchor point
      P_obs = segment_coords_abs[obs_len-1,:]
      P_shifted = segment_coords_abs - P_obs


      # feature calculation and augumentation

      P_displacement = P_shifted[1:,:] - P_shifted[:-1,:]
      displacement_padding = np.zeros((1,2))
      P_displacement = np.vstack([displacement_padding,P_displacement])

      # rotation for the shifted possition and the displacement

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


  return all_segments




class Data_Loader:
  def __init__(self,segments:List[Dict[str,Any]],args : Any):
    self.segments = segments
    self.args = args
    self.obs_len = args.obs_len
    self.pred_len = args.pred_len
    self.tot_len = args.total_len
    self.neighbour_threshold = args.neighbour_threshold

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
    P_shifted = torch.from_numpy(seg['obs_coords_shifted']).float()
    abs_s[:,i,:] = P_shifted[:t]
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
 


