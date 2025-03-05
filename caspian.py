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
    dec_dim = 1024  # Decoder LSTM units and prejoction
    attn_dim = 128  # Attention projection size
    conv_channels = 32  # For location convolution
    kernel_size = 31  # For location convolution
    prenet_dim = 256  # Pre-net hidden units
    postnet_filters = 512  # Post-net filters
    postnet_kernel_size = 5  # Post-net kernel size
    mel_dim = 80  # Mel spectrogram dimension (example value)
    query_attention_dim = 256  # Decoder hidden size (for query)




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
    def __init__(self, enc_dim, query_attention_dim = 256, attn_dim=128, conv_channels=32, kernel_size=31):
        super(LocationSensitiveAttention, self).__init__()
        self.W_query = nn.Linear(query_attention_dim, attn_dim, bias=False)
        self.W_key = nn.Linear(enc_dim, attn_dim, bias=False)
        self.conv_location = nn.Conv1d(1, conv_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.W_location = nn.Linear(conv_channels, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

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

# Tacotron 2 Decoder
class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        # Pre-net: 2 FC layers with 256 units
        self.prenet = nn.Sequential(
            nn.Linear(hparams.mel_dim, hparams.prenet_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  # Tacotron typically uses dropout
            nn.Linear(hparams.prenet_dim, hparams.prenet_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Attention
        self.attention = LocationSensitiveAttention(
            hparams.enc_blstm_hidden_size, hparams.query_attention_dim,
            hparams.attn_dim, hparams.conv_channels, hparams.kernel_size
        )
        # 2-layer uni-directional LSTM with 1024 units
        self.lstm = nn.LSTM(
            hparams.prenet_dim + hparams.enc_blstm_hidden_size,  # Concatenated input size
            hparams.dec_dim,
            num_layers=2,
            batch_first=True
        )
        # Projection to mel spectrogram
        self.mel_projection = nn.Linear(
            hparams.dec_dim + hparams.enc_blstm_hidden_size,  # Concatenated LSTM output + context
            hparams.mel_dim
        )
        # Post-net: 5-layer conv with 512 filters, shape 5x1
        self.postnet = nn.Sequential(
            nn.Conv1d(hparams.mel_dim, hparams.postnet_filters, hparams.postnet_kernel_size, padding=2),
            nn.BatchNorm1d(hparams.postnet_filters),
            nn.Tanh(),
            nn.Conv1d(hparams.postnet_filters, hparams.postnet_filters, hparams.postnet_kernel_size, padding=2),
            nn.BatchNorm1d(hparams.postnet_filters),
            nn.Tanh(),
            nn.Conv1d(hparams.postnet_filters, hparams.postnet_filters, hparams.postnet_kernel_size, padding=2),
            nn.BatchNorm1d(hparams.postnet_filters),
            nn.Tanh(),
            nn.Conv1d(hparams.postnet_filters, hparams.postnet_filters, hparams.postnet_kernel_size, padding=2),
            nn.BatchNorm1d(hparams.postnet_filters),
            nn.Tanh(),
            nn.Conv1d(hparams.postnet_filters, hparams.mel_dim, hparams.postnet_kernel_size, padding=2)
        )

    def forward(self, encoder_outputs, mel_targets=None, max_steps=10):
        """
        Args:
            encoder_outputs: (batch_size, seq_len, enc_blstm_hidden_size)
            mel_targets: (batch_size, mel_dim, max_target_len) for teacher forcing, optional
            max_steps: Max decoding steps if no targets provided
        Returns:
            mel_outputs: (batch_size, mel_dim, max_len)
            mel_outputs_post: (batch_size, mel_dim, max_len)
            attention_weights_all: (batch_size, max_len, seq_len)
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        # Initialize decoder input (first frame is zero or from targets)
        if mel_targets is not None:
            decoder_input = torch.zeros(batch_size, hparams.mel_dim, device=device).unsqueeze(2)  # (batch, mel_dim, 1)
            max_len = mel_targets.size(2)
        else:
            decoder_input = torch.zeros(batch_size, hparams.mel_dim, device=device).unsqueeze(2)
            max_len = max_steps

        # Initialize attention and LSTM state
        prev_attention = torch.zeros(batch_size, encoder_outputs.size(1), device=device)
        lstm_state = None
        mel_outputs = []
        attention_weights_all = []

        for t in range(max_len):
            # Pre-net
            prenet_out = self.prenet(decoder_input.squeeze(2))  # (batch, prenet_dim)

            # Attention
            if t == 0:
                decoder_hidden = torch.zeros(batch_size, hparams.dec_dim, device=device)
            attention_weights = self.attention(encoder_outputs, decoder_hidden, prev_attention)
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, enc_dim)
            context = context.squeeze(1)  # (batch, enc_dim)

            # LSTM
            lstm_input = torch.cat([prenet_out, context], dim=1).unsqueeze(1)  # (batch, 1, prenet_dim + enc_dim)
            lstm_out, lstm_state = self.lstm(lstm_input, lstm_state)  # (batch, 1, dec_dim)
            decoder_hidden = lstm_out.squeeze(1)  # (batch, dec_dim)

            # Mel projection
            mel_input = torch.cat([lstm_out.squeeze(1), context], dim=1)  # (batch, dec_dim + enc_dim)
            mel_out = self.mel_projection(mel_input).unsqueeze(2)  # (batch, mel_dim, 1)

            # Update for next step
            mel_outputs.append(mel_out)
            attention_weights_all.append(attention_weights)
            prev_attention = attention_weights
            decoder_input = mel_targets[:, :, t].unsqueeze(2) if mel_targets is not None else mel_out

        # Stack outputs
        mel_outputs = torch.cat(mel_outputs, dim=2)  # (batch, mel_dim, max_len)
        attention_weights_all = torch.stack(attention_weights_all, dim=1)  # (batch, max_len, seq_len)

        # Post-net
        mel_outputs_post = self.postnet(mel_outputs) + mel_outputs  # (batch, mel_dim, max_len)

        return mel_outputs, mel_outputs_post, attention_weights_all

# Example Usage
hparams = HParams()
batch_size, seq_len, mel_len = 2, 10, 20

# Initialize models
encoder = Encoder(hparams)
decoder = Decoder(hparams)

# Simulated input
inputs = torch.randint(0, hparams.num_vocab, (batch_size, seq_len))  # (batch, seq_len)
mel_targets = torch.randn(batch_size, hparams.mel_dim, mel_len)  # Dummy mel spectrogram

# Forward pass
encoder_outputs = encoder(inputs)  # (batch, seq_len, enc_blstm_hidden_size)
mel_outputs, mel_outputs_post, attention_weights = decoder(encoder_outputs, mel_targets)

print("Encoder output shape:", encoder_outputs.shape)
print("Mel outputs shape:", mel_outputs.shape)
print("Mel outputs post-net shape:", mel_outputs_post.shape)
print("Attention weights shape:", attention_weights.shape)