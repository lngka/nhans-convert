import tensorflow as tf
if int(tf.version.VERSION.split('.')[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()
import numpy as np
import librosa
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy import signal

############################################# Global variables ###########################################
SAMPLING_RATE = 16000
WIN_SAMPLE = int(SAMPLING_RATE * 0.025)
HOP_SAMPLE = int(SAMPLING_RATE * 0.010)
MIX_WIN = 35
NOISE_WIN = 200

############################################# Helpers ####################################################
def read_wav(in_path):
    rate, samples = wavread(in_path)
    assert rate == SAMPLING_RATE
    assert samples.dtype == 'int16'
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)
    assert len(samples.shape) == 1
    return samples


def handle_signals(mixedpath, noisepospath, noisenegpath):
    try:
        # Read Wavs
        mixedsamples = read_wav(mixedpath)
        noisepossamples = read_wav(noisepospath)
        noisenegsamples = read_wav(noisenegpath)

        # Normalize
        mixedsamples = mixedsamples / (max(abs(mixedsamples))+0.000001)
        noisepossamples = noisepossamples / (max(abs(noisepossamples))+0.000001)
        noisenegsamples = noisenegsamples / (max(abs(noisenegsamples))+0.000001)
        mixedsamples = mixedsamples.astype(np.float32)
        noisepossamples = noisepossamples.astype(np.float32)
        noisenegsamples = noisenegsamples.astype(np.float32)

        # Cut the end to have an exact number of frames
        win_samples = int(SAMPLING_RATE * 0.025)
        hop_samples = int(SAMPLING_RATE * 0.010)
        if (len(mixedsamples) - win_samples) % hop_samples != 0:
            mixedsamples = mixedsamples[:-((len(mixedsamples) - win_samples) % hop_samples)]

        print('================================handle_signals===========================================')
        print('SAMPLING_RATE: ', SAMPLING_RATE)
        print('win_samples', win_samples)
        print('hop_samples', hop_samples)
        print('noisepossamples: ', noisepossamples.shape)
        print('noisenegsamples', noisenegsamples.shape)
        print('mixedsamples', mixedsamples.shape)
        print('number of frames',
              (1 + (len(mixedsamples) - win_samples) / hop_samples))

        return mixedsamples, noisepossamples, noisenegsamples
    except:
        print('error in threads')
        print(mixedpath, noisepospath, noisenegpath)



def pad_and_strided_crop(tensor, length, stride):
    # we assume that we have a length dimension and a feature dimension
    assert len(tensor.shape) == 2
    n_features = int(tensor.shape[1])

    len_before = ((length + 1) // 2) - 1
    len_after = length // 2
    padded = tf.pad(tensor, [[len_before, len_after], [0, 0]])

    expanded = tf.expand_dims(tf.expand_dims(padded, axis=0), axis=3)

    windows = tf.extract_image_patches(expanded,
                                       ksizes=[1, length, n_features, 1],
                                       strides=[1, stride, n_features, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')
    return tf.reshape(windows, [-1, length, n_features])


def apply_nhans_tflite(mixedpath, pospath, negpath, save_to):
    # data processing
    mix_wav, noise_pos_wav, noise_neg_wav = handle_signals(mixedpath, pospath, negpath)

    # noise_pos_wav, noise_neg_wav, mix_wav = [tf.reshape(x, [-1]) for x in (noise_pos_wav, noise_neg_wav, mix_wav)]
    # print('================================reshape=================================================')
    # print('noise_pos_wav: ', noise_pos_wav.shape)
    # print('noise_neg_wav', noise_neg_wav.shape)
    # print('mix_wav', mix_wav.shape)

    win_samples = int(SAMPLING_RATE * 0.025)
    hop_samples = int(SAMPLING_RATE * 0.010)
    mix_stft = tf.signal.stft(mix_wav, frame_length=win_samples, frame_step=hop_samples, fft_length=win_samples)
    pos_stft = tf.signal.stft(noise_pos_wav, win_samples, hop_samples, win_samples)
    neg_stft = tf.signal.stft(noise_neg_wav, win_samples, hop_samples, win_samples)
    # mix_stft = librosa.stft(mix_wav, n_fft=400, hop_length=160, window=400, win_length=400, dtype=np.float32)
    # pos_stft = librosa.stft(noise_pos_wav, n_fft=400, hop_length=160, window=400, win_length=400, dtype=np.float32)
    # neg_stft = librosa.stft(noise_neg_wav, n_fft=400, hop_length=160, window=400, win_length=400, dtype=np.float32)
    print('================================STFT=====================================================')
    print('mix_stft: ', mix_stft.shape)
    print('pos_stft: ', pos_stft.shape)
    print('neg_stft: ', neg_stft.shape)


    mix_phase = tf.angle(mix_stft)
    mix_spectrum = tf.log(tf.abs(mix_stft) + 0.00001)
    pos_spectrum = tf.log(tf.abs(pos_stft) + 0.00001)
    neg_spectrum = tf.log(tf.abs(neg_stft) + 0.00001)
    print('================================Spectrum=================================================')
    print('mix_phase: ', mix_phase)
    print('mix_spectrum: ', mix_spectrum)
    print('pos_spectrum: ', pos_spectrum)
    print('neg_spectrum: ', neg_spectrum)

    mix_spectra = pad_and_strided_crop(mix_spectrum, MIX_WIN, 1)
    print('================================Spectra==================================================')
    print('mix_spectra: ', mix_spectra)

    pos_spectrum = pos_spectrum[:NOISE_WIN]
    pos_spectrum = tf.reshape(pos_spectrum, [NOISE_WIN, pos_spectrum.shape[1].value])
    pos_expanded = tf.expand_dims(pos_spectrum, 0)
    pos_spectra = tf.tile(pos_expanded, [tf.shape(mix_spectra)[0], 1, 1])
    print('pos_spectra: ', pos_spectra)

    neg_spectrum = neg_spectrum[:NOISE_WIN]
    neg_spectrum = tf.reshape(neg_spectrum, [NOISE_WIN, neg_spectrum.shape[1].value])
    pos_expanded = tf.expand_dims(neg_spectrum, 0)
    neg_spectra = tf.tile(pos_expanded, [tf.shape(mix_spectra)[0], 1, 1])
    print('neg_spectra: ', neg_spectra)

    return


def apply_denoiser(mixedpath, negpath, save_to):
    dir = './audio_examples/'
    pospath = dir + 'Silent.wav'
    apply_nhans_tflite(mixedpath, pospath, negpath, save_to)


if __name__ == '__main__':
    mixedpath = './audio_examples/mixed.wav'
    negpath = './audio_examples/noise.wav'
    save_to = './audio_examples/denoised.wav'

    apply_denoiser(mixedpath, negpath, save_to)
