import torch
from basemodel import Temperal_Encoder

# Dummy args container
class DummyArgs:
    input_mix = False           # use 2 input channels
    hidden_size = 64
    x_encoder_head = 4
    x_encoder_layers = 2

    # ---- TCN SETTINGS ----
    use_tcn = True
    tcn_layers = 3
    tcn_hidden = 64
    tcn_kernel = 3
    tcn_dropout = 0.0

    # unused args but required by basemodel
    # add placeholders to avoid errors
    ifGaussian = False
    mlp_decoder = False
    input_offset = True
    input_position = False
    pass

# Create encoder
args = DummyArgs()
encoder = Temperal_Encoder(args)

# Dummy input (batch=5, channels=2, seq_len=8)
x = torch.randn(5, 2, 8)

print("Input shape:", x.shape)

# Forward pass
out, state, cn = encoder(x)

print("Encoder output shape:", out.shape)
print("State shape:", state.shape)
print("CN shape:", cn.shape)

print("\nTCN UNIT TEST COMPLETE ✔")
