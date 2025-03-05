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
    dec_lstm_dim = 1024  # Decoder LSTM units
    attn_dim = 128  # Attention projection size
    attention_conv_channels = 32  # For location convolution
    attention_kernel_size = 31  # For location convolution
    prenet_dim = 256  # Pre-net hidden units
    postnet_filters = 512  # Post-net filters
    postnet_kernel_size = 5  # Post-net kernel size
    mel_dim = 80  # Mel spectrogram dimension (example value)
    query_attention_dim = 1024  # Decoder hidden size (for query), updated to match dec_lstm_dim

    
    #Audio
    #Audio parameters are the most important parameters to tune when using this work on your personal data. Below are the beginner steps to adapt
    #this work to your personal data:
    #	1- Determine my data sample rate: First you need to determine your audio sample_rate (how many samples are in a second of audio). This can be done using sox: "sox --i <filename>"
    #		(For this small tuto, I will consider 24kHz (24000 Hz), and defaults are 22050Hz, so there are plenty of examples to refer to)
    #	2- set sample_rate parameter to your data correct sample rate
    #	3- Fix win_size and and hop_size accordingly: (Supposing you will follow our advice: 50ms window_size, and 12.5ms frame_shift(hop_size))
    #		a- win_size = 0.05 * sample_rate. In the tuto example, 0.05 * 24000 = 1200
    #		b- hop_size = 0.25 * win_size. Also equal to 0.0125 * sample_rate. In the tuto example, 0.25 * 1200 = 0.0125 * 24000 = 300 (Can set frame_shift_ms=12.5 instead)
    #	4- Fix n_fft, num_freq and upsample_scales parameters accordingly.
    #		a- n_fft can be either equal to win_size or the first power of 2 that comes after win_size. I usually recommend using the latter
    #			to be more consistent with signal processing friends. No big difference to be seen however. For the tuto example: n_fft = 2048 = 2**11
    #		b- num_freq = (n_fft / 2) + 1. For the tuto example: num_freq = 2048 / 2 + 1 = 1024 + 1 = 1025.
    #		c- For WaveNet, upsample_scales products must be equal to hop_size. For the tuto example: upsample_scales=[15, 20] where 15 * 20 = 300
    #			it is also possible to use upsample_scales=[3, 4, 5, 5] instead. One must only keep in mind that upsample_kernel_size[0] = 2*upsample_scales[0]
    #			so the training segments should be long enough (2.8~3x upsample_scales[0] * hop_size or longer) so that the first kernel size can see the middle 
    #			of the samples efficiently. The length of WaveNet training segments is under the parameter "max_time_steps".
    #	5- Finally comes the silence trimming. This very much data dependent, so I suggest trying preprocessing (or part of it, ctrl-C to stop), then use the
    #		.ipynb provided in the repo to listen to some inverted mel/linear spectrograms. That will first give you some idea about your above parameters, and
    #		it will also give you an idea about trimming. If silences persist, try reducing trim_top_db slowly. If samples are trimmed mid words, try increasing it.
    #	6- If audio quality is too metallic or fragmented (or if linear spectrogram plots are showing black silent regions on top), then restart from step 2.
    num_mels = 80 #Number of mel-spectrogram channels and local conditioning dimensionality
    num_freq = 1025# (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
    rescale = True #Whether to rescale audio prior to preprocessing
    rescaling_max = 0.999 #Rescaling value

    #train samples of lengths between 3sec and 14sec are more than enough to make a model capable of generating consistent speech.
    clip_mels_length = True #For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, also consider clipping your samples to smaller chunks)
    max_mel_frames = 900  #Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.

    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It's preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws=False #Only used to set as True if using WaveNet, no difference in performance is observed in either cases.
    silence_threshold=2 #silence threshold used for sound trimming for wavenet preprocessing

    #Mel spectrogram
    n_fft = 2048 #Extra window size is filled with 0 paddings to match this parameter
    hop_size = 275 #For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
    win_size = 1100#For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate = 22050 #22050 Hz (corresponding to ljspeech dataset) (sox --i <filename>)
    frame_shift_ms = None, #Can replace hop_size parameter. (Recommended: 12.5)
    magnitude_power = 2. #The power of the spectrogram magnitude (1. for energy, 2. for power)

    #M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
    trim_silence = True #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    trim_fft_size = 2048 #Trimming window size
    trim_hop_size = 512 #Trimmin hop length
    trim_top_db = 40 #Trimming db difference from reference db (smaller==harder trim.)

    #Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization = True  #Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization = True  #Only relevant if mel_normalization = True
    symmetric_mels = True  #Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
    max_abs_value = 4.  #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, 
                                                                                                            #not too small for fast convergence)
    normalize_for_wavenet = True  #whether to rescale to [0, 1] for wavenet. (better audio quality)
    clip_for_wavenet = True #whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)
    wavenet_pad_sides = 1  #Can be 1 or 2. 1 for pad right only, 2 for both sides padding.

    #Contribution by @begeekmyfriend
    #Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
    preemphasize = True  #whether to apply filter
    preemphasis = 0.97  #filter coefficient.

    #Limits
    min_level_db = -100
    ref_level_db = 20
    fmin = 55 #Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax = 7600 #To be increased/reduced depending on data.

    #Griffin Lim
    power = 1.5 #Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
    griffin_lim_iters = 60 #Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
    GL_on_GPU = True #Whether to use G&L GPU version as part of tensorflow graph. (Usually much faster than CPU but slightly worse quality too).
    ###########################################################################################################################################
