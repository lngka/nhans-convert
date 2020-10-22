import math
import time
import tensorflow as tf
if int(tf.version.VERSION.split('.')[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()
import numpy as np, functools
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

def preprocess(mixedpath, noisepospath, noisenegpath):
    try:
        mixedsamples = read_wav(mixedpath)
        noisepossamples = read_wav(noisepospath)
        noisenegsamples = read_wav(noisenegpath)

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

        print('================================Wave===========================================')
        print('SAMPLING_RATE: ', SAMPLING_RATE)
        print('win_samples', win_samples)
        print('hop_samples', hop_samples)
        print('noisepossamples: ', noisepossamples.shape)
        print('noisenegsamples', noisenegsamples.shape)
        print('mixedsamples', mixedsamples.shape)
        print('number of frames',
              (1 + (len(mixedsamples) - win_samples) / hop_samples))
        
        tf.compat.v1.enable_eager_execution()

        # data processing
        mix_wav = mixedsamples
        noise_pos_wav = noisepossamples
        noise_neg_wav = noisenegsamples

        noise_pos_wav, noise_neg_wav, mix_wav = [tf.reshape(x, [-1]) for x in (noise_pos_wav, noise_neg_wav, mix_wav)]
        # print('================================reshape=================================================')
        # print('noise_pos_wav: ', noise_pos_wav.shape)
        # print('noise_neg_wav', noise_neg_wav.shape)
        # print('mix_wav', mix_wav.shape)

        win_samples = int(SAMPLING_RATE * 0.025)
        hop_samples = int(SAMPLING_RATE * 0.010)
        mix_stft = tf.signal.stft(mix_wav, frame_length=win_samples, frame_step=hop_samples, fft_length=win_samples)
        pos_stft = tf.signal.stft(noise_pos_wav, win_samples, hop_samples, win_samples)
        neg_stft = tf.signal.stft(noise_neg_wav, win_samples, hop_samples, win_samples)

        print('================================STFT=====================================================')
        print('mix_stft: ', mix_stft.shape)
        print('pos_stft: ', pos_stft.shape)
        print('neg_stft: ', neg_stft.shape)


        mix_phase = tf.angle(mix_stft)
        mix_spectrum = tf.log(tf.abs(mix_stft) + 0.00001)
        pos_spectrum = tf.log(tf.abs(pos_stft) + 0.00001)
        neg_spectrum = tf.log(tf.abs(neg_stft) + 0.00001)
        print('================================Spectrum=================================================')
        print('mix_phase: ', mix_phase.shape)
        print('mix_spectrum: ', mix_spectrum.shape)
        print('pos_spectrum: ', pos_spectrum.shape)
        print('neg_spectrum: ', neg_spectrum.shape)

        mix_spectra = pad_and_strided_crop(mix_spectrum, MIX_WIN, 1)
        print('================================Spectra==================================================')
        print('mix_spectra: ', mix_spectra.shape)

        pos_spectrum = pos_spectrum[:NOISE_WIN]
        pos_spectrum = tf.reshape(pos_spectrum, [NOISE_WIN, pos_spectrum.shape[1].value])
        pos_expanded = tf.expand_dims(pos_spectrum, 0)
        pos_spectra = tf.tile(pos_expanded, [tf.shape(mix_spectra)[0], 1, 1])
        print('pos_spectra: ', pos_spectra.shape)

        if (neg_spectrum.shape[0] < NOISE_WIN) :
            mul = math.ceil(NOISE_WIN / neg_spectrum.shape[0].value)
            neg_spectrum = tf.tile(neg_spectrum, [mul, 1])
        
        neg_spectrum = neg_spectrum[:NOISE_WIN]
        neg_spectrum = tf.reshape(neg_spectrum, [NOISE_WIN, neg_spectrum.shape[1].value])
        neg_expanded = tf.expand_dims(neg_spectrum, 0)
        neg_spectra = tf.tile(neg_expanded, [tf.shape(mix_spectra)[0], 1, 1])
        print('neg_spectra: ', neg_spectra.shape)

        return mix_phase, mix_spectra, pos_spectra, neg_spectra
    except:
        print('error in handle signal')
        print(mixedpath, noisepospath, noisenegpath)

def recover_samples_from_spectrum(logspectrum_stft, spectrum_phase, save_to):
    abs_spectrum = np.exp(logspectrum_stft)
    spectrum_phase = np.array(spectrum_phase)
    spectrum = abs_spectrum * (np.exp(1j * spectrum_phase))

    istft_graph = tf.Graph()
    with istft_graph.as_default():
        num_fea = int(int(SAMPLING_RATE * 0.025) / 2 + 1)
        frame_length = int(SAMPLING_RATE * 0.025)
        frame_step = int(SAMPLING_RATE * 0.010)
        stft_ph = tf.placeholder(tf.complex64, shape=(None, num_fea))
        samples = tf.signal.inverse_stft(stft_ph, frame_length, frame_step, frame_length, window_fn=tf.signal.inverse_stft_window_fn(frame_step, forward_window_fn=functools.partial(tf.signal.hann_window, periodic=True)))
        istft_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        samples_ = istft_sess.run(samples, feed_dict={stft_ph: spectrum})
        wavwrite(save_to, SAMPLING_RATE, samples_)

    return samples_

def timelog(start_time = None):
    if start_time == None:
        start_time = time.time() # start time
        return start_time
    else:
        end = time.time()
        print("\nDone in ", (end - start_time) / 60, 'minutes')

def print_denoised():
    denoised = np.load('./denoised.npy')
    print(denoised)
    


############################################# main ####################################################
def denoise(mixedpath, pospath, negpath):
    mix_phase, mix_spectra, pos_spectra, neg_spectra = preprocess(mixedpath, pospath, negpath)

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="./tflite/n_hans.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print('================================Inputs/Outputs of TFLite Model============================')
    mix_spectra_index = input_details[0]['index']
    neg_spectra_index = input_details[1]['index']
    pos_spectra_index = input_details[2]['index']
    out_spectra_index = output_details[0]['index']


    print(f'mixedph {mix_spectra_index}:  ', input_details[0])
    print(f'noisenegcontextph {neg_spectra_index}:  ', input_details[1])
    print(f'noiseposcontextph {pos_spectra_index}:  ', input_details[2])
    print(f'add_72 {out_spectra_index}:  ', output_details[0])


    # Set input
    interpreter.set_tensor(neg_spectra_index, tf.slice(neg_spectra, begin=[0, 0, 0], size=[1, 200, 201]))
    interpreter.set_tensor(pos_spectra_index, tf.slice(pos_spectra, begin=[0, 0, 0], size=[1, 200, 201]))

    
    print('================================Start denoising============================================')
    start_time = timelog()
    denoised = [] # list of denoised tensor
    num_frames = mix_spectra.shape[0]
    for i in range(num_frames):
        interpreter.set_tensor(mix_spectra_index, tf.slice(mix_spectra, begin=[i, 0, 0], size=[1, 35, 201]))
        interpreter.invoke()

        output_data = interpreter.get_tensor(out_spectra_index)
        print(output_data.shape)
        denoised.append(output_data)
        print(f"Progress: {i+1}/{num_frames}", end="\r")

    # print elapsed time, save np array
    timelog(start_time)
    denoised = np.concatenate(denoised, axis=0)
    print(f'add_72 {denoised}:  ', denoised.shape)
    np.save('./denoised.npy', denoised)

    return mix_phase, denoised


if __name__ == '__main__':
    mixedpath = './audio_examples/mixed.wav'
    negpath = './audio_examples/noise.wav'
    pospath = './audio_examples/Silent.wav'
    save_to = './audio_examples/denoised.wav'
    #print_denoised()
    mix_phase, denoised = denoise(mixedpath, pospath, negpath)
    recover_samples_from_spectrum(mix_phase, denoised, save_to)
