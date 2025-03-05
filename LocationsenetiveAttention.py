import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
class HParams:
    num_vocab = 30
    enc_emb_dim = 512
    enc_num_conv_layers = 3
    enc_conv_channels = 512
    enc_conv_kernel_size = 5
    tacotron_dropout_rate = 0.5
    enc_blstm_hidden_size = 512
    enc_blstm_num_layers = 1
    dec_dim = 256  # Decoder hidden size (for query)
    attn_dim = 128  # Attention projection size (as per essay)
    conv_channels = 32  # For location convolution (as per essay)
    kernel_size = 31  # For location convolution (as per essay)

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

# Location-Sensitive Attention (Updated per Essay)
class LocationSensitiveAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim=128, conv_channels=32, kernel_size=31):
        super(LocationSensitiveAttention, self).__init__()
        self.W_query = nn.Linear(dec_dim, attn_dim, bias=False)  # Project to 128-dim
        self.W_key = nn.Linear(enc_dim, attn_dim, bias=False)    # Project to 128-dim
        self.conv_location = nn.Conv1d(1, conv_channels, kernel_size, padding=(kernel_size - 1) // 2)  # 32 filters, length 31
        self.W_location = nn.Linear(conv_channels, attn_dim, bias=False)  # Project to 128-dim
        self.v = nn.Linear(attn_dim, 1, bias=False)  # Final scoring

    def forward(self, encoder_outputs, decoder_hidden, prev_attention):
        """
        Args:
            encoder_outputs: (batch_size, seq_len, enc_dim)
            decoder_hidden: (batch_size, dec_dim)
            prev_attention: (batch_size, seq_len)
        Returns:
            attention_weights: (batch_size, seq_len)
        """
        query = self.W_query(decoder_hidden).unsqueeze(1)  # (batch_size, 1, 128)
        keys = self.W_key(encoder_outputs)                 # (batch_size, seq_len, 128)
        prev_attention = prev_attention.unsqueeze(1)       # (batch_size, 1, seq_len)
        location_features = self.conv_location(prev_attention)  # (batch_size, 32, seq_len)
        location_features = location_features.transpose(1, 2)   # (batch_size, seq_len, 32)
        location_term = self.W_location(location_features)      # (batch_size, seq_len, 128)
        energy = self.v(torch.tanh(keys + query + location_term)).squeeze(-1)  # (batch_size, seq_len)
        attention_weights = F.softmax(energy, dim=1)  # (batch_size, seq_len)
        return attention_weights

# Simple Decoder Stub
class DecoderStub(nn.Module):
    def __init__(self, hparams):
        super(DecoderStub, self).__init__()
        self.lstm = nn.LSTM(hparams.dec_dim, hparams.dec_dim, batch_first=True)
        self.attention = LocationSensitiveAttention(
            hparams.enc_blstm_hidden_size, hparams.dec_dim,
            hparams.attn_dim, hparams.conv_channels, hparams.kernel_size
        )

    def forward(self, encoder_outputs, decoder_input, prev_attention):
        output, _ = self.lstm(decoder_input)  # (batch, 1, dec_dim)
        attention_weights = self.attention(encoder_outputs, output.squeeze(1), prev_attention)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, enc_dim)
        return context, attention_weights

# Example Usage
hparams = HParams()
batch_size, seq_len = 2, 10

# Initialize models
encoder = Encoder(hparams)
decoder = DecoderStub(hparams)

# Simulated input
inputs = torch.randint(0, hparams.num_vocab, (batch_size, seq_len))  # (batch, seq_len)
decoder_input = torch.randn(batch_size, 1, hparams.dec_dim)  # Dummy decoder input
prev_attention = torch.zeros(batch_size, seq_len)  # Initial attention weights

# Forward pass
encoder_outputs = encoder(inputs)  # (batch, seq_len, enc_blstm_hidden_size)
print("Encoder output shape:", encoder_outputs.shape)

context, attention_weights = decoder(encoder_outputs, decoder_input, prev_attention)
print("Context shape:", context.shape)  # (batch, 1, enc_blstm_hidden_size)
print("Attention weights shape:", attention_weights.shape)  # (batch, seq_len)
print("Sample attention weights:", attention_weights[0])