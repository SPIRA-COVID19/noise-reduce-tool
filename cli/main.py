import io
import os
import math as m
import numpy as np
from urllib.request import urlopen
import scipy.io.wavfile as wavfile
import ctypes
import soundfile as sf
import librosa
from common.noisereduce import reduce_noise


def load_file(filename):
    return librosa.load(filename)


def preprocess_audio(y, sr):
    """
        Preprocess an audio file: centralizes audio at y=0 and adjusts
        amplitude to [-1, 1].
    """
    # Librosa already adjusts it to us.
    # sample_depth = 8*y[0].itemsize
    y = y[:] - np.mean(y)
    # y = y / (2**(sample_depth - 1))


def sliding_window_energy(y, sr, window_size=4096):
    """
        Calculates the mean energy (in dB) of the signal in sliding windows.
        returns the mean energy edB and its minimum value.
    """
    y2 = np.power(y, 2)
    window = np.ones(window_size) / float(window_size)

    # decibel calculation.
    # Some borders of the convolution may have zeroes on them, and that
    # makes taking log10 especially hard. We'll ignore them and leave them zero.
    convolution = np.convolve(y2, window)
    edB = 10 * np.log10(convolution, where=convolution > 0)[window_size - 1:]

    # we throw away the initial and ending 0.5s, because the sliding windows
    # are not correct in the initial/final borders.
    imin = int(0.5 * sr)

    edBmin = np.min(edB[imin:-imin])
    edBmax = np.max(edB[imin:-imin])
    edB = np.maximum(edB, edBmin)

    return edB, edBmin, edBmax


def boolean_majority_filter(y, window_size):
    """
        Applies a majority filter boolean vectors
        over windows of size 2 * window_size + 1 
    """
    y_out = y.copy()

    # we initalize the sentry as N True values. This makes the edges of the sound be considered as
    # noise more often, which is more common anyway.
    y_pad = np.concatenate(
        (np.ones(window_size), y, np.ones(window_size + 1)))

    n_true = 0
    n_false = 0

    for i in range(2 * window_size + 1):
        n_true += int(y_pad[i])
        n_false += int(not y_pad[i])

    for i in range(len(y_out)):
        # Every index is the majority vote of the window y[i-window_size : i+window_size + 1]
        # containing 2 * window_size + 1 elements.
        # that corresponds to indexes y_pad[i:i + 2 * window_size + 1]
        y_out[i] = n_true > n_false

        if i >= window_size:
            to_remove = y_out[i - window_size]
        else:  # remove one "True" that we padded.
            to_remove = y_pad[i]

        if to_remove:
            n_true -= 1
        else:
            n_false -= 1

        # gets the new vote and includes it.
        if y_pad[i + 2 * window_size + 1]:
            n_true += 1
        else:
            n_false += 1

    return y_out


def noise_sel(y, sr, noise_threshold: float = None, eliminate_noise_bigger_than_seconds: float = 0.2):
    edB, edBmin, edBmax = sliding_window_energy(y, sr)

    if noise_threshold is None:
        # Crude heuristic: anything below 27% of the "mean dB"
        # is considered noise.
        noise_threshold = 0.27 * (edBmax - edBmin)


    # select frames with RMS mean next to the minimum level
    inoise_pre = edB < edBmin + noise_threshold

    inoise = boolean_majority_filter(inoise_pre, int(
        eliminate_noise_bigger_than_seconds * sr))

    return inoise, inoise_pre

def cut_noise_from_edges(y, inoise):
    '''
    Cuts all the noise from the beginning and end of the signal.
    this is made using the indices of inoise.
    '''
    # noise = True, signal = False. It returns rows and columns, we just want the rows.
    isignal, *_ = np.where(inoise == False)
    first_signal, last_signal = isignal[0], isignal[-1]

    return y[first_signal:last_signal]

def noise_reduce_signal(y, sr):
    # We can only work with audios longer than 1 second, because
    # we will throw away at least 0.5s off of each side
    if len(y) <= sr * 1:
        return y, np.zeros(len(y))

    inoise, _ = noise_sel(y, sr)
    noise = y[inoise]

    reduced_y, ε = reduce_noise(audio_clip=y,
                                noise_clip=noise, 
                                n_grad_freq=3,
                                n_grad_time=3, 
                                n_std_thresh=2, 
                                prop_decrease=1.0, 
                                verbose=False)

    reduced_y = cut_noise_from_edges(reduced_y, inoise)

    # Normalize to [-1, 1]
    reduced_y /= max(max(y),-min(y),1)
    ε /= max(max(ε),-min(ε),1)

    return reduced_y, ε

def just_crop_ends(y, sr):
    if len(y) <= sr * 1:
        return y
    
    inoise, _ = noise_sel(y, sr)
    return cut_noise_from_edges(y, inoise)


def process_signal_file(filename, save_to, noise_supress=True):
    y, sr = load_file(filename)
    if noise_supress:
        reduced_y, _ = noise_reduce_signal(y, sr)
    else:
        reduced_y = just_crop_ends(y, sr)
    sf.write(save_to, reduced_y, sr)

def main(argv):
    from pathlib import Path
    from os import makedirs

    IGNORED_PATHS = ['.DS_Store', '.asd']

    if len(argv) < 3:
        print(f'usage: {argv[0]} <location_to_be_saved> <file/dir> [ <file/dir> ... ]')
        print(f'Cleans all noise from audio in all file/dirs in the input.')
        return -1

    # I'll make it in a cleaner way afterwards
    no_noise_supression = False
    if '--no-noise-supression' in argv:
        index_of_no_noise_supression_flag = argv.index('--no-noise-supression')
        no_noise_supression = True
        argv.pop(index_of_no_noise_supression_flag)

    output_path = argv[1].rstrip('/')
    makedirs(output_path, exist_ok=True)

    for search_path in argv[2:]:
        if Path(search_path).is_file():
            just_name = str(Path(search_path).relative_to(Path(search_path).parent)).split('.')[0]
            makedirs(Path(output_path) / Path(search_path).relative_to(search_path).parent, exist_ok=True)
            process_signal_file(search_path, f'{output_path}/{just_name}.cleaned.wav')
            print(f'processed {search_path}')
            continue
        for path in Path(search_path).rglob('*'):
            if any(x in str(path) for x in IGNORED_PATHS):
                continue
            if not path.is_file():
                continue
            just_name = str(path.relative_to(search_path)).split('.')[0]
            makedirs(Path(output_path) / Path(path).relative_to(search_path).parent, exist_ok=True)
            process_signal_file(path, f'{output_path}/{just_name}.cleaned.wav')
            print(f'processed {path}')

    

if __name__ == '__main__':
    from sys import argv, exit
    exit(main(argv))