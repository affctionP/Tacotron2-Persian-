

import torch
import torch.nn as nn


# Define Tacotron 2 Components
class Encoder(nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.embeddingLayer = nn.Embedding(hparams.num_vocab, hparams.enc_emb_dim)
        # Define convolutional layers as a module list
        self.convLayers = nn.ModuleList([
            EncoderConvolutions(hparams.enc_emb_dim, hparams)
            for _ in range(hparams.enc_num_conv_layers)
        ])

        # Bi-directional LSTM
        self.blstm = nn.LSTM(hparams.enc_conv_channels, hparams.enc_blstm_hidden_size//2, hparams.enc_blstm_num_layers,
                             batch_first=True, bidirectional=True)

    def forward(self, x):
        # Apply embedding layer
        x = self.embeddingLayer(x).transpose(1, 2)  # Shape: (batch, emb_dim, seq_len)
        
        # Pass through each convolutional layer
        for conv_layer in self.convLayers:
            x = conv_layer(x)

        # x = pack_padded_sequence(x.transpose(1,2), x_lens, batch_first=True)
        #(batch_size=, seq_len=, input_size=) lstem expect shape like this 
        x=x.transpose(1,2)
        x,_=self.blstm(x)
        return x


class EncoderConvolutions(nn.Module):
    """Encoder convolutional layers used to find local dependencies in input characters."""
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
class HParams:
    num_vocab = 30
    enc_emb_dim = 512
    enc_num_conv_layers = 3
    enc_conv_channels = 512
    enc_conv_kernel_size = 5
    tacotron_dropout_rate = 0.5
    enc_blstm_hidden_size=512
    enc_blstm_num_layers=1

# Example usage
hparams = HParams()
encoder = Encoder( hparams=hparams)

# Simulated input (batch_size=2, seq_len=10)
inputs = torch.randint(0, hparams.num_vocab, (2, 10))  # Random integer tokens
outputs = encoder(inputs)

print("Output shape:", outputs.shape)  # Expected: (2, enc_conv_channels, seq_len)
