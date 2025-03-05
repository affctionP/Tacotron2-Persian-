

# import torch
# import torch.nn as nn


# # Define Tacotron 2 Components
# class Encoder(nn.Module):
#     def __init__(self, hparams):
#         super(Encoder, self).__init__()
#         self.embeddingLayer = nn.Embedding(hparams.num_vocab, hparams.enc_emb_dim)
#         # Define convolutional layers as a module list
#         self.convLayers = nn.ModuleList([
#             EncoderConvolutions(hparams.enc_emb_dim, hparams)
#             for _ in range(hparams.enc_num_conv_layers)
#         ])

#         # Bi-directional LSTM
#         self.blstm = nn.LSTM(hparams.enc_conv_channels, hparams.enc_blstm_hidden_size//2, hparams.enc_blstm_num_layers,
#                              batch_first=True, bidirectional=True)

#     def forward(self, x):
#         # Apply embedding layer
#         x = self.embeddingLayer(x).transpose(1, 2)  # Shape: (batch, emb_dim, seq_len)
        
#         # Pass through each convolutional layer
#         for conv_layer in self.convLayers:
#             x = conv_layer(x)

#         # x = pack_padded_sequence(x.transpose(1,2), x_lens, batch_first=True)
#         #(batch_size=, seq_len=, input_size=) lstem expect shape like this 
#         x=x.transpose(1,2)
#         x,_=self.blstm(x)
#         return x


# class EncoderConvolutions(nn.Module):
#     """Encoder convolutional layers used to find local dependencies in input characters."""
#     def __init__(self, inp_channels, hparams):
#         super(EncoderConvolutions, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(inp_channels, hparams.enc_conv_channels, hparams.enc_conv_kernel_size,
#                       stride=1, padding=(hparams.enc_conv_kernel_size - 1) // 2, bias=False),
#             nn.BatchNorm1d(hparams.enc_conv_channels),
#             nn.ReLU(),
#             nn.Dropout(hparams.tacotron_dropout_rate)
#         )

#     def forward(self, x):
#         return self.conv(x)
# class HParams:
#     num_vocab = 30
#     enc_emb_dim = 512
#     enc_num_conv_layers = 3
#     enc_conv_channels = 512
#     enc_conv_kernel_size = 5
#     tacotron_dropout_rate = 0.5
#     enc_blstm_hidden_size=512
#     enc_blstm_num_layers=1

# # Example usage
# hparams = HParams()
# encoder = Encoder( hparams=hparams)

# # Simulated input (batch_size=2, seq_len=10)
# inputs = torch.randint(0, hparams.num_vocab, (2, 10))  # Random integer tokens
# outputs = encoder(inputs)

# print("Output shape:", outputs.shape)  # Expected: (2, enc_conv_channels, seq_len)
import torch
import torch.nn as nn
import torch.nn.functional as F



# Tacotron 2 Encoder
class Encoder(nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.embeddingLayer = nn.Embedding(hparams.num_vocab, hparams.enc_emb_dim)
        self.convLayers = nn.ModuleList([
            EncoderConvolutions(hparams.enc_emb_dim, hparams)
            for _ in range(hparams.enc_num_conv_layers)
        ])
        self.blstm = nn.LSTM(
            hparams.enc_conv_channels,
            hparams.enc_blstm_hidden_size // 2,
            hparams.enc_blstm_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        x = self.embeddingLayer(x).transpose(1, 2)  # (batch, emb_dim, seq_len)
        for conv_layer in self.convLayers:
            x = conv_layer(x)
        x = x.transpose(1, 2)  # (batch, seq_len, enc_conv_channels)
        x, _ = self.blstm(x)  # (batch, seq_len, enc_blstm_hidden_size)
        return x

class EncoderConvolutions(nn.Module):
    def __init__(self, inp_channels, hparams):
        super(EncoderConvolutions, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(inp_channels, hparams.enc_conv_channels, hparams.enc_conv_kernel_size,
                      stride=1, padding=(hparams.enc_conv_kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(hparams.enc_conv_channels),
            nn.ReLU(),
            nn.Dropout(hparams.tacotron_dropout_rate)
        )

    def forward(self, x):
        return self.conv(x)

# Location-Sensitive Attention
class LocationSensitiveAttention(nn.Module):


    def __init__(self, enc_dim, query_attention_dim, attn_dim=128, attention_conv_channels=32, attention_kernel_size=31):
        super(LocationSensitiveAttention, self).__init__()
        self.W_query = nn.Linear(query_attention_dim, attn_dim, bias=False)  # Project to 128-dim
        self.W_key = nn.Linear(enc_dim, attn_dim, bias=False)    # Project to 128-dim
        self.conv_location = nn.Conv1d(1, attention_conv_channels, attention_kernel_size, padding=(attention_kernel_size - 1) // 2)  # 32 filters, length 31
        self.W_location = nn.Linear(attention_conv_channels, attn_dim, bias=False)  # Project to 128-dim
        self.v = nn.Linear(attn_dim, 1, bias=False)  # Final scoring

    def forward(self, encoder_outputs, decoder_hidden, prev_attention):
        query = self.W_query(decoder_hidden).unsqueeze(1)  # (batch_size, 1, 128)
        keys = self.W_key(encoder_outputs)                 # (batch_size, seq_len, 128)
        prev_attention = prev_attention.unsqueeze(1)       # (batch_size, 1, seq_len)
        location_features = self.conv_location(prev_attention)  # (batch_size, 32, seq_len)
        location_features = location_features.transpose(1, 2)   # (batch_size, seq_len, 32)
        location_term = self.W_location(location_features)      # (batch_size, seq_len, 128)
        energy = self.v(torch.tanh(keys + query + location_term)).squeeze(-1)  # (batch_size, seq_len)
        attention_weights = F.softmax(energy, dim=1)  # (batch_size, seq_len)
        return attention_weights
