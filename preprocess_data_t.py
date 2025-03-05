import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import librosa
import librosa.filters
import numpy as np
import audio_process as audio
from hparams import HParams
# from wavenet_vocoder.util import is_mulaw, is_mulaw_quantize, mulaw, mulaw_quantize


def build_from_path(hparams, input_dirs, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
    """
    Preprocesses the speech dataset from a gven input path to given output directories

    Args:
        - hparams: hyper parameters
        - input_dir: input directory that contains the files to prerocess audioes and meta.csv
        - mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
        - linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
        - wav_dir: output directory of the preprocessed speech audio dataset
        - n_jobs: Optional, number of worker process to parallelize across
        - tqdm: Optional, provides a nice progress bar

    Returns:
        - A list of tuple describing the train examples. this should be written to train.txt
    """

    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1
    for input_dir in input_dirs:
        with open(os.path.join(input_dir, 'test.csv'), encoding='utf-8') as f:
            lines = f.readlines()  # Read all lines into a list
            for line in lines[1:]:
                parts = line.strip().split(',')
                basename = parts[1]
                wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format("00"+basename))
                text = parts[2]
                futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams)))
                index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text, hparams):
    """
    Preprocesses a single utterance wav/text pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
        - mel_dir: the directory to write the mel spectograms into
        - linear_dir: the directory to write the linear spectrograms into
        - wav_dir: the directory to write the preprocessed wav into
        - index: the numeric index to use in the spectogram filename
        - wav_path: path to the audio file containing the speech input
        - text: text spoken in the input audio file
        - hparams: hyper parameters

    Returns:
        - A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
    """
    try:
        # Load the audio as numpy array
        
        wav = audio.load_wav(wav_path, sr=hparams.sample_rate)	
        
    except FileNotFoundError: #catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
            wav_path))
        return None

    #Trim lead/trail silences
    # if hparams.trim_silence:
    # 	wav = audio.trim_silence(wav, hparams)

    #Pre-emphasize
    preem_wav = audio.preemphasis(wav, hparams.preemphasis, hparams.preemphasize)

    #rescale wav
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
        preem_wav = preem_wav / np.abs(preem_wav).max() * hparams.rescaling_max

        #Assert all audio is in [-1, 1]
        if (wav > 1.).any() or (wav < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(wav_path))
        if (preem_wav > 1.).any() or (preem_wav < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(wav_path))

    #Mu-law quantize
    # if is_mulaw_quantize(hparams.input_type):
    # 	#[0, quantize_channels)
    # 	out = mulaw_quantize(wav, hparams.quantize_channels)

    # 	#Trim silences
    # 	start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
    # 	wav = wav[start: end]
    # 	preem_wav = preem_wav[start: end]
    # 	out = out[start: end]

    # 	constant_values = mulaw_quantize(0, hparams.quantize_channels)
    # 	out_dtype = np.int16

    # elif is_mulaw(hparams.input_type):
    # 	#[-1, 1]
    # 	out = mulaw(wav, hparams.quantize_channels)
    # 	constant_values = mulaw(0., hparams.quantize_channels)
    # 	out_dtype = np.float32

    # else:
        #[-1, 1]
    out = wav
    constant_values = 0.
    out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(preem_wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    #Compute the linear scale spectrogram from the wav
    linear_spectrogram = audio.linearspectrogram(preem_wav, hparams).astype(np.float32)
    linear_frames = linear_spectrogram.shape[1]

    #sanity check
    assert linear_frames == mel_frames

    if hparams.use_lws:
        #Ensure time resolution adjustement between audio and mel-spectrogram
        fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
        l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

        #Zero pad audio signal
        out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
    else:
        #Ensure time resolution adjustement between audio and mel-spectrogram
        l_pad, r_pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams), hparams.wavenet_pad_sides)

        #Reflect pad audio signal on the right (Just like it's done in Librosa to avoid frame inconsistency)
        out = np.pad(out, (l_pad, r_pad), mode='constant', constant_values=constant_values)

    assert len(out) >= mel_frames * audio.get_hop_size(hparams)

    #time resolution adjustement
    #ensure length of raw audio is multiple of hop size so that we can use
    #transposed convolution to upsample
    out = out[:mel_frames * audio.get_hop_size(hparams)]
    assert len(out) % audio.get_hop_size(hparams) == 0
    time_steps = len(out)

    # Write the spectrogram and audio to disk
    audio_filename = 'audio-{}.npy'.format(index)
    mel_filename = 'mel-{}.npy'.format(index)
    linear_filename = 'linear-{}.npy'.format(index)
    np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example
    return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text)




def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	mel_frames = sum([int(m[4]) for m in metadata])
	timesteps = sum([int(m[3]) for m in metadata])
	sr = hparams_1.sample_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), mel_frames, timesteps, hours))
	print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
	print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))


hparams_1=HParams()
print(hparams_1.sample_rate)
input_dirs=['./Dataset/']
import os

# Define directory paths
mel_dir = './training_dir/mels/'
linear_dir = './training_dir/linear/'
wav_dir = './training_dir/edit_wavs/'

# List of directories to check/create
directories = [mel_dir, linear_dir, wav_dir]

# Check and create each directory if it doesn't exist
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print(f"Directory already exists: {directory}")
meta_data=build_from_path(hparams_1, input_dirs, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x)
write_metadata(meta_data,'./training_dir/')