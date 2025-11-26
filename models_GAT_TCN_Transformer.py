import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Utility: Positional Encoding
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x):
        # x shape: [seq_len, batch, d_model] or [batch, seq_len, d_model]
        if x.dim() == 3 and x.shape[0] <= self.pe.shape[0]:
            # assume [seq_len, batch, d_model]
            seq_len = x.shape[0]
            return x + self.pe[:seq_len].unsqueeze(1).to(x.device)
        elif x.dim() == 3:
            # maybe [batch, seq_len, d_model]
            seq_len = x.shape[1]
            return x + self.pe[:seq_len].unsqueeze(0).to(x.device)
        else:
            return x

# ---------------------------
# Spatial GAT Implementation
# ---------------------------
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, concat=True, dropout=0.0, alpha=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.concat = concat
        self.W = nn.Linear(in_dim, out_dim, bias=False)       # Wh
        self.a = nn.Linear(2 * out_dim, 1, bias=False)        # attention vector
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj_mask):
        """
        h: [N, in_dim]
        adj_mask: [N, N] bool (1 neighbor, 0 not neighbor)
        returns: [N, out_dim]
        """
        device = h.device
        N = h.size(0)
        Wh = self.W(h)  # [N, out_dim]

        # prepare pairwise combinations efficiently:
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1)  # [N, N, out_dim]
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1)  # [N, N, out_dim]
        cat = torch.cat([Wh_i, Wh_j], dim=-1)     # [N, N, 2*out_dim]

        e = self.leakyrelu(self.a(cat).squeeze(-1))  # [N, N]

        # mask out non-neighbors
        if adj_mask is None:
            mask = torch.ones_like(e, dtype=torch.bool, device=device)
        else:
            mask = adj_mask.to(torch.bool)

        # ensure self-loop present
        diag_idx = torch.arange(0, N, device=device)
        mask[diag_idx, diag_idx] = True

        neg_inf = -9e15
        e_masked = e.masked_fill(~mask, neg_inf)

        alpha = torch.softmax(e_masked, dim=1)  # normalize over j for each i -> [N, N]
        alpha = self.dropout(alpha)

        h_prime = torch.matmul(alpha, Wh)  # [N, out_dim]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=2, dropout=0.0, alpha=0.2, last_layer=False):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.heads = nn.ModuleList([
            GATLayer(in_dim, self.head_dim, concat=(not last_layer), dropout=dropout, alpha=alpha)
            for _ in range(num_heads)
        ])
        self.last_layer = last_layer

    def forward(self, h, adj_mask):
        head_outs = [head(h, adj_mask) for head in self.heads]  # each [N, head_dim]
        # Always concatenate heads to get out_dim
        out = torch.cat(head_outs, dim=-1)  # [N, out_dim = num_heads * head_dim]
        return out

class SpatialGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=2, dropout=0.0):
        super().__init__()
        # first layer: concat heads -> hidden_dim
        self.gat1 = MultiHeadGAT(in_dim, hidden_dim, num_heads=num_heads, dropout=dropout, last_layer=False)
        # second layer: average heads -> out_dim
        self.gat2 = MultiHeadGAT(hidden_dim, out_dim, num_heads=num_heads, dropout=dropout, last_layer=True)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h, adj_mask):
        # h: [N, in_dim]
        x = self.gat1(h, adj_mask)  # [N, hidden_dim]
        x = self.gat2(x, adj_mask)  # [N, out_dim]
        x = self.norm(x)
        return x  # [N, out_dim]

# ---------------------------
# Temporal TCN (small, residual)
# ---------------------------
class TemporalTCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, num_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            dilation = 1
            padding = (kernel_size - 1) // 2  # keep sequence length same
            conv = nn.Conv1d(in_ch, hidden_channels, kernel_size, padding=padding, dilation=dilation)
            layers.append(nn.Sequential(conv, nn.ReLU(), nn.Dropout(dropout)))
        self.net = nn.Sequential(*layers)
        # final projection (optional)
        self.res_proj = nn.Conv1d(in_channels, hidden_channels, 1) if in_channels != hidden_channels else None

    def forward(self, x):
        """
        x: [batch=N, channels, seq_len]  (we treat N=agents as batch)
        returns: [batch=N, hidden_channels, seq_len]
        """
        res = self.res_proj(x) if self.res_proj is not None else x
        out = self.net(x)
        return out + res  # residual

# ---------------------------
# Transformer Decoder for trajectories
# ---------------------------
class TransformerTrajectoryDecoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=4, pred_len=12, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model, max_len=pred_len+50)
        # head to produce mean (mu) and scale (sigma)
        self.mu_head = nn.Linear(d_model, 2)
        self.sigma_head = nn.Linear(d_model, 2)
        self.min_scale = 1e-3

    def forward(self, memory):
        """
        memory: [mem_len, batch=N, d_model] (mem_len = obs_len)
        returns:
            loc: [pred_len, N, 2]
            scale: [pred_len, N, 2]  (positive)
        """
        device = memory.device
        N = memory.shape[1]
        # prepare tgt as zeros with positional enc
        tgt = torch.zeros(self.pred_len, N, self.d_model, device=device)  # [pred_len, N, d_model]
        tgt = self.pos_enc(tgt)
        mem = memory  # already assumed pos-encoded by caller if required
        out = self.transformer_decoder(tgt, mem)  # [pred_len, N, d_model]
        loc = self.mu_head(out)  # [pred_len, N, 2]
        sigma_raw = self.sigma_head(out)  # [pred_len, N, 2]
        sigma = F.softplus(sigma_raw) + self.min_scale
        return loc, sigma

# ---------------------------
# Gaussian NLL Loss (bivariate independent dims)
# ---------------------------
class GaussianNLLLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred: concatenated loc and scale OR a tuple (loc, scale)
        - loc: [pred_len, N, 2]
        - scale: [pred_len, N, 2] (positive)
        target: [pred_len, N, 2]
        return: scalar loss
        """
        if isinstance(pred, tuple) or isinstance(pred, list):
            loc, scale = pred
        else:
            # assume last dim concatenation
            raise ValueError("pred should be (loc, scale) tuple")
        # compute elementwise nll assuming independent dims:
        # nll = 0.5 * (log(2*pi) + 2*log(scale) + ((y - loc)^2) / scale^2)
        scale = scale.clone()
        scale = scale.clamp_min(self.eps)
        diff2 = (target - loc) ** 2
        nll = 0.5 * (torch.log(2 * math.pi * (scale ** 2)) + diff2 / (scale ** 2))
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll

# ---------------------------
# Top-level model wrapper
# ---------------------------
class GAT_TCN_Transformer(nn.Module):
    """
    Wrapper that replicates the old GATraj forward signature:
    forward(inputs, epoch, iftest=False)
    where inputs = (batch_abs_gt, batch_norm_gt, nei_list_batch, nei_num_batch, batch_split)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden = getattr(args, "hidden_size", 64)
        self.hidden_size = hidden
        self.obs_len = args.obs_length
        self.pred_len = args.pred_length

        # Spatial GAT: input dim is 2 (x,y) unless you change to include velocities
        self.spat_gat = SpatialGAT(in_dim=2, hidden_dim=hidden, out_dim=hidden, num_heads=2, dropout=0.0)

        # TCN: we use agent as batch dimension, channels = hidden, seq_len = obs_len
        self.tcn = TemporalTCN(in_channels=hidden, hidden_channels=hidden, kernel_size=getattr(args, "tcn_kernel", 3),
                               num_layers=getattr(args, "tcn_layers", 2), dropout=getattr(args, "tcn_dropout", 0.0))

        # a light linear to map tcn features to transformer d_model (hidden)
        self.memory_proj = nn.Linear(hidden, hidden)

        # positional encoding for memory (obs frames)
        self.mem_pos_enc = PositionalEncoding(hidden, max_len=self.obs_len+50)

        # Transformer decoder
        self.decoder = TransformerTrajectoryDecoder(d_model=hidden, nhead=getattr(args, "transformer_heads", 4),
                                                    num_layers=getattr(args, "transformer_layers", 4),
                                                    pred_len=self.pred_len, dropout=0.1)

        # loss
        self.reg_loss = GaussianNLLLoss(reduction='mean')

    # helper to build a full adjacency mask across the concatenated batch
    def build_adj_mask_for_frame(self, nei_list_batch, batch_split, t, device):
        """
        nei_list_batch: list length = num_scenes_in_minibatch
           each element: [H, N_scene, N_scene] (numpy or list)
        batch_split: list of [left, right] pairs indicating index range in full batch
        t: time index (0..obs_len-1)
        returns: [N_total, N_total] bool mask
        """
        # compute total agents:
        total_agents = 0
        for (l, r) in batch_split:
            total_agents += (r - l)
        adj = torch.zeros((total_agents, total_agents), dtype=torch.bool, device=device)
        # fill blocks
        for b_idx, (lr) in enumerate(batch_split):
            left, right = lr[0], lr[1] if isinstance(lr, (list, tuple)) else (lr[0], lr[1])
            # in original code batch_split elements are [start, end]
            left, right = lr[0], lr[1] if isinstance(lr, (list, tuple)) else (lr[0], lr[1])
        # Above handling is defensive; simpler:
        for b_idx, br in enumerate(batch_split):
            left, right = br[0], br[1]
            # nei_list_batch[b_idx] is either a list or numpy array: shape [H, N_scene, N_scene]
            nei_scene = nei_list_batch[b_idx]
            # ensure it's tensor
            if isinstance(nei_scene, list):
                nei_scene = torch.tensor(nei_scene, device=device)
            else:
                nei_scene = torch.tensor(nei_scene, device=device) if not torch.is_tensor(nei_scene) else nei_scene.to(device)
            # some datasets store nei_list for full seq_length, with indices aligned; pick frame t
            # clip t to available length if needed
            if t >= nei_scene.shape[0]:
                t_idx = nei_scene.shape[0] - 1
            else:
                t_idx = t
            block = nei_scene[t_idx].bool()  # [N_scene, N_scene]
            adj[left:right, left:right] = block
        # ensure diagonal self loops
        idx = torch.arange(0, adj.shape[0], device=device)
        adj[idx, idx] = True
        return adj

    def forward(self, inputs, epoch=None, iftest=False):
        """
        inputs: tuple:
          batch_abs_gt: [H, N, 2]
          batch_norm_gt: [H, N, 2]
          nei_list_batch: list (per scene) each [H, N_scene, N_scene]
          nei_num_batch: [N, H] or similar (unused here)
          batch_split: list of [left, right] ranges
        returns:
          loss, [best_prediction_trajectories_list]
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_abs_gt, batch_norm_gt, nei_list_batch, nei_num_batch, batch_split = inputs
        # Ensure tensors
        batch_norm_gt = batch_norm_gt.to(device)
        # observed frames positions
        obs_pos = batch_norm_gt[:self.obs_len, :, :2]  # [obs_len, N, 2]
        H, N, _ = obs_pos.shape

        # collect spatial embeddings per frame
        spatial_emb_list = []
        for t in range(self.obs_len):
            # node features: [N, 2]
            h_t = obs_pos[t].to(device)  # [N, 2]
            # build adjacency mask for frame t
            adj_mask_t = self.build_adj_mask_for_frame(nei_list_batch, batch_split, t, device)  # [N, N]
            # Spatial GAT expects float features; adj boolean mask
            gat_out = self.spat_gat(h_t.float(), adj_mask_t)  # [N, hidden]
            spatial_emb_list.append(gat_out.unsqueeze(0))  # [1, N, hidden]

        # stack: [obs_len, N, hidden]
        spatial_emb = torch.cat(spatial_emb_list, dim=0)
        # transform to [N, hidden, obs_len] for TCN (batch=agents)
        spatial_emb_batch = spatial_emb.permute(1, 2, 0)  # [N, hidden, obs_len]

        # run TCN (agent-as-batch)
        tcn_out = self.tcn(spatial_emb_batch)  # [N, hidden, obs_len]
        # project hidden -> d_model (same here)
        tcn_out = tcn_out.permute(2, 0, 1)  # [obs_len, N, hidden]
        # optional positional encoding on memory
        tcn_out = self.mem_pos_enc(tcn_out)  # [obs_len, N, hidden]

        # decode with Transformer decoder
        loc, scale = self.decoder(tcn_out)  # both [pred_len, N, 2]

        # compute loss against ground truth normalized positions
        # predicted y should be compared to batch_norm_gt[self.obs_len:, :, :2] shape [pred_len, N, 2]
        target = batch_norm_gt[self.obs_len:, :, :2].to(device)  # [pred_len, N, 2]
        # ensure shapes match
        if target.shape[0] != loc.shape[0]:
            # handle mismatch: crop/pad target or loc
            L = min(target.shape[0], loc.shape[0])
            target = target[:L]
            loc = loc[:L]
            scale = scale[:L]

        loss = self.reg_loss((loc, scale), target)

        # Build full predicted trajectory(s) for evaluation
        # The evaluation expects: obs frames [1:obs_len] + pred frames = total 19 frames
        # obs frames 1 to 7 (7 frames) + pred frames 8-19 (12 frames) = 19 frames
        pre_obs = batch_norm_gt[1:self.obs_len, :, :2].to(device)  # [obs_len-1, N, 2] = [7, N, 2]
        pred_cumsum = torch.cumsum(loc, dim=0)  # [pred_len, N, 2] = [12, N, 2]
        
        full_pre_tra = []
        # Concatenate observed trajectory (from frame 1) with predictions
        full_traj = torch.cat([pre_obs, pred_cumsum], dim=0)  # [7+12, N, 2] = [19, N, 2]
        full_pre_tra.append(full_traj)

        return loss, full_pre_tra

# For backward compatibility naming:
GAT_TCN_Transformer = GAT_TCN_Transformer
