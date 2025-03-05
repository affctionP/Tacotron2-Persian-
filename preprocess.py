import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import pandas as pd
import os
from pathlib import Path
import librosa

# Assuming your Encoder, Decoder, and HParams classes are defined as in previous messages
# Here's a minimal HParams for reference (adjust as needed)
class HParams:
    num_vocab = 148  # Size of character vocabulary (adjust based on your tokenizer)
    enc_emb_dim = 512
    enc_num_conv_layers = 3
    enc_conv_channels = 512
    enc_conv_kernel_size = 5
    tacotron_dropout_rate = 0.5
    enc_blstm_hidden_size = 512
    enc_blstm_num_layers = 1
    dec_lstm_dim = 1024
    attn_dim = 128
    attention_conv_channels = 32
    attention_kernel_size = 31
    prenet_dim = 256
    postnet_filters = 512
    postnet_kernel_size = 5
    mel_dim = 80
    query_attention_dim = 1024
    sample_rate = 22050  # LJSpeech sample rate
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    n_mels = 80

# Text Preprocessing (Simple Character-Level Tokenizer)
def text_to_sequence(text, char_to_idx):
    return [char_to_idx.get(char, 0) for char in text.lower()]  # 0 for unknown chars

def build_vocab():
    # Basic character set (adjust based on your dataset)
    chars = "abcdefghijklmnopqrstuvwxyz .,!?-'" + "".join(str(i) for i in range(10))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}  # 0 reserved for padding/unknown
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

# Audio Preprocessing (Mel Spectrogram)
def audio_to_mel(audio_path, hparams):
    y, sr = librosa.load(audio_path, sr=hparams.sample_rate)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=hparams.n_fft, hop_length=hparams.hop_length, 
        win_length=hparams.win_length, n_mels=hparams.n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return torch.tensor(mel_spec_db, dtype=torch.float32)  # Shape: (n_mels, time)

# Custom Dataset for LJSpeech
class LJspeechDataset(Dataset):
    def __init__(self, data_dir, char_to_idx, hparams, max_seq_len=200, max_mel_len=1000):
        self.data_dir = data_dir
        self.char_to_idx = char_to_idx
        self.hparams = hparams
        self.max_seq_len = max_seq_len
        self.max_mel_len = max_mel_len
        
        # Load metadata
        metadata_path = os.path.join(data_dir, "metadata.csv")
        self.metadata = pd.read_csv(metadata_path, sep="|", header=None, names=["id", "text", "normalized_text"])
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        wav_id = self.metadata.iloc[idx]["id"]
        text = self.metadata.iloc[idx]["normalized_text"] or self.metadata.iloc[idx]["text"]
        
        # Text to sequence
        text_seq = text_to_sequence(text, self.char_to_idx)
        text_seq = text_seq[:self.max_seq_len]  # Truncate if too long
        text_seq = text_seq + [0] * (self.max_seq_len - len(text_seq))  # Pad with 0s
        text_tensor = torch.tensor(text_seq, dtype=torch.long)
        
        # Audio to mel spectrogram
        wav_path = os.path.join(self.data_dir, "wavs", f"{wav_id}.wav")
        mel_spec = audio_to_mel(wav_path, self.hparams)  # (n_mels, time)
        mel_spec = mel_spec[:, :self.max_mel_len]  # Truncate if too long
        if mel_spec.shape[1] < self.max_mel_len:
            mel_spec = torch.nn.functional.pad(mel_spec, (0, self.max_mel_len - mel_spec.shape[1]))
        
        return text_tensor, mel_spec

# Checkpoint Functions (from previous response)
def save_checkpoint(epoch, encoder, decoder, optimizer, loss, checkpoint_dir="checkpoints"):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_latest_checkpoint(encoder, decoder, optimizer, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return 0
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    if not checkpoints:
        return 0
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[2].split(".")[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    last_loss = checkpoint['loss']
    print(f"Loaded checkpoint: {checkpoint_path}, resuming from epoch {start_epoch}, last loss: {last_loss:.4f}")
    return start_epoch

# Training Function
def train(encoder, decoder, train_loader, epochs=10, learning_rate=0.001, device='cuda', checkpoint_dir="checkpoints", save_interval=1):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    start_epoch = load_latest_checkpoint(encoder, decoder, optimizer, checkpoint_dir)
    
    encoder.train()
    decoder.train()
    
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        for batch_idx, (text_inputs, mel_targets) in enumerate(train_loader):
            text_inputs = text_inputs.to(device)  # (batch_size, max_seq_len)
            mel_targets = mel_targets.to(device)  # (batch_size, mel_dim, max_mel_len)
            
            stop_targets = torch.zeros(mel_targets.size(0), mel_targets.size(2), 1, device=device)
            stop_targets[:, -1, :] = 1
            
            optimizer.zero_grad()
            
            encoder_outputs = encoder(text_inputs)
            mel_outputs, mel_outputs_post, attention_weights, stop_tokens = decoder(encoder_outputs, mel_targets)
            
            mel_loss_pre = mse_loss(mel_outputs, mel_targets)
            mel_loss_post = mse_loss(mel_outputs_post, mel_targets)
            stop_loss = bce_loss(stop_tokens.squeeze(-1), stop_targets.squeeze(-1))
            loss = mel_loss_pre + mel_loss_post + stop_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Mel Pre: {mel_loss_pre.item():.4f}, '
                      f'Mel Post: {mel_loss_post.item():.4f}, Stop: {stop_loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}] completed, Average Loss: {avg_loss:.4f}')
        
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            save_checkpoint(epoch, encoder, decoder, optimizer, avg_loss, checkpoint_dir)
    
    return encoder, decoder

# Main Execution
if __name__ == "__main__":
    # Hyperparameters
    hparams = HParams()
    
    # Build vocabulary
    char_to_idx, idx_to_char = build_vocab()
    hparams.num_vocab = len(char_to_idx) + 1  # +1 for padding/unknown
    
    # Initialize models
    encoder = Encoder(hparams)
    decoder = Decoder(hparams)
    
    # Dataset and DataLoader
    data_dir = "path/to/LJSpeech-1.1"  # Replace with your LJSpeech directory
    dataset = LJspeechDataset(data_dir, char_to_idx, hparams, max_seq_len=200, max_mel_len=1000)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    
    # Train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder, decoder = train(
        encoder,
        decoder,
        train_loader,
        epochs=50,  # Adjust based on your needs
        learning_rate=0.001,
        device=device,
        checkpoint_dir="checkpoints",
        save_interval=5  # Save every 5 epochs
    )