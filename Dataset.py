from torch.utils.data import Dataset,DataLoader
from audio_process import audio_to_mel
import pandas as pd
import os 
import torch
import sys
from g2p.g2p import Grapheme2Phoneme

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

class HParams :
    sample_rate = 16000  # Common Voice uses 16kHz
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    n_mels = 80
class PersianTTSDataset(Dataset):
    def __init__(self, data_dir,phonemizer_obj , hparams, max_seq_len=200, max_mel_len=1000):
        self.data_dir = data_dir
        self.hparams = hparams
        self.max_seq_len = max_seq_len
        self.max_mel_len = max_mel_len
        self.phonemizer_obj=phonemizer_obj

        self.metadata = pd.read_csv(os.path.join(data_dir, "test.csv"))
        self.metadata = self.metadata[["filename", "text"]]  # Keep only path and text
        self.clips_dir = os.path.join(data_dir, "wavs")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        text = row["text"]
        audio_path = os.path.join(self.clips_dir, "00"+str(row["filename"])+".wav") #add zero pading
      
        # Text to sequence
        text_seq =self.phonemizer_obj.text_to_phone(text)
        text_seq = self.phonemizer_obj.text_to_sequence(text_seq)
        #text_seq = text_seq + [0] * (self.max_seq_len - len(text_seq))
        print(f"len text is {len(text_seq)}")
            
        text_padded = pad_sequence(text_seq, self.max_seq_len)  # Define max_len as needed
        #audio_padded = pad_sequence(audio, max_len=500)
        text_tensor = torch.tensor(text_padded, dtype=torch.long)

        
        # Audio to mel spectrogram
        mel_spec = audio_to_mel(audio_path, self.hparams)
        mel_spec = mel_spec[:, :self.max_mel_len]
        if mel_spec.shape[1] < self.max_mel_len:
            mel_spec = torch.nn.functional.pad(mel_spec, (0, self.max_mel_len - mel_spec.shape[1]))
        
        return text_tensor, mel_spec


def pad_sequence(sequence, max_len):
    if len(sequence) < max_len:
        padding = torch.zeros(max_len - len(sequence))
        return torch.cat((sequence, padding))
    return sequence[:max_len]
    


hparams = HParams()


gpobject=Grapheme2Phoneme()
hparams.num_vocab = len(gpobject.char_list)

# Dataset (Common Voice Persian)
data_dir = "./Dataset/"  # Replace with your extracted Common Voice folder
dataset = PersianTTSDataset(data_dir,gpobject, hparams, max_seq_len=500, max_mel_len=1000)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

phonem=gpobject.text_to_phone("چاه عمیق آب گندیده")
# print(gpobject.phone_to_sequence(phonem))
# print (phonem)
for text, labels in train_loader:
    print("Images:", text)
    print("Labels:", labels)
    break  # Remove this if you want to print all batches
