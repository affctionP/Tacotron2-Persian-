import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path
from decoder import Decoder,Encoder
from hparams import HParams

def save_checkpoint(epoch, encoder, decoder, optimizer, loss, checkpoint_dir="checkpoints"):
    """Save model and optimizer state to a checkpoint file."""
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
    """Load the latest checkpoint if it exists, return the epoch to resume from."""
    if not os.path.exists(checkpoint_dir):
        return 0  # No checkpoint exists, start from epoch 0
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    if not checkpoints:
        return 0  # No valid checkpoints found
    
    # Find the latest epoch
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[2].split(".")[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    
    # Load states
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    last_loss = checkpoint['loss']
    
    print(f"Loaded checkpoint: {checkpoint_path}, resuming from epoch {start_epoch}, last loss: {last_loss:.4f}")
    return start_epoch

def train(encoder, decoder, train_loader, epochs=10, learning_rate=0.001, device='cuda', checkpoint_dir="checkpoints", save_interval=1):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # Optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    # Load latest checkpoint if available
    start_epoch = load_latest_checkpoint(encoder, decoder, optimizer, checkpoint_dir)
    
    encoder.train()
    decoder.train()
    
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        for batch_idx, (text_inputs, mel_targets) in enumerate(train_loader):
            text_inputs = text_inputs.to(device)
            mel_targets = mel_targets.to(device)
            
            # Stop token targets
            stop_targets = torch.zeros(mel_targets.size(0), mel_targets.size(2), 1, device=device)
            stop_targets[:, -1, :] = 1
            
            optimizer.zero_grad()
            
            # Forward pass
            encoder_outputs = encoder(text_inputs)
            mel_outputs, mel_outputs_post, attention_weights, stop_tokens = decoder(encoder_outputs, mel_targets)
            
            # Compute losses
            mel_loss_pre = mse_loss(mel_outputs, mel_targets)
            mel_loss_post = mse_loss(mel_outputs_post, mel_targets)
            stop_loss = bce_loss(stop_tokens.squeeze(-1), stop_targets.squeeze(-1))
            loss = mel_loss_pre + mel_loss_post + stop_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Mel Pre: {mel_loss_pre.item():.4f}, '
                      f'Mel Post: {mel_loss_post.item():.4f}, Stop: {stop_loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}] completed, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint at specified interval or at the last epoch
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            save_checkpoint(epoch, encoder, decoder, optimizer, avg_loss, checkpoint_dir)
    
    return encoder, decoder

# Example usage
from torch.utils.data import DataLoader, TensorDataset

hparams = HParams()
encoder = Encoder(hparams)
decoder = Decoder(hparams)

# Dummy dataset
batch_size, seq_len, mel_len = 32, 10, 20
text_inputs = torch.randint(0, hparams.num_vocab, (100, seq_len))
mel_targets = torch.randn(100, hparams.mel_dim, mel_len)
dataset = TensorDataset(text_inputs, mel_targets)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train with checkpointing
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder, decoder = train(
    encoder, 
    decoder, 
    train_loader, 
    epochs=10, 
    learning_rate=0.001, 
    device=device, 
    checkpoint_dir="checkpoints", 
    save_interval=1  # Save every 2 epochs
)