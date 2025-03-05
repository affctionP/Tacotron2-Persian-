import librosa
import torch 
import numpy as np

def audio_to_mel(audio_path, hparams):
    y, sr = librosa.load(audio_path, sr=hparams.sample_rate)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=hparams.n_fft, hop_length=hparams.hop_length,
        win_length=hparams.win_length, n_mels=hparams.n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return torch.tensor(mel_spec_db, dtype=torch.float32)