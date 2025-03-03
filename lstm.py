import torch
import torch.nn as nn

# Define LSTM layer
lstm = nn.LSTM(
    input_size=10,    # 10 features in each input step
    hidden_size=20,   # 20 features in hidden state
    num_layers=2,     # 2 stacked LSTM layers
    bias=True,        # Include bias terms
    batch_first=True, # Input shape: (batch, seq_len, input_size)
    dropout=0.1,      # 10% dropout
    bidirectional=True, # Bidirectional LSTM
)

# Example input
input_tensor = torch.randn(32, 5, 10)  # (batch_size=32, seq_len=5, input_size=10)

# Forward pass
output, (hn, cn) = lstm(input_tensor)
print(f"output shape {output.shape}")
print(f"output shape hn {hn.shape}")
print(f"output shape hn {cn.shape}")