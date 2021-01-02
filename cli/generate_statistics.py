import numpy as np
from dataclasses import dataclass, fields, asdict
from pathlib import Path
import librosa
from concurrent.futures import Future, wait
from functools import partial
import sys

from common import process_directory_raw, NoiseSuppressor

def count_sizes(is_noise):
    '''
    counts the sizes of the signals, splitting on each
    noise skip.
    '''
    isignal = np.where(is_noise == False)[0]
    expected = isignal[0] + 1
    current_signal_size = 0
    signal_sizes = []
    for i in isignal[1:]:
        current_signal_size += 1
        if i != expected:
            signal_sizes.append(current_signal_size)
            current_signal_size = 0
        expected = i + 1
    signal_sizes.append(current_signal_size)
    return signal_sizes

@dataclass
class Statistics:
    filename: str
    noise_voice_ratio: float
    noise_ratio: float
    amount_of_skips: int
    signal_length_avg: float
    signal_length_stddev: float

def generate_statistics_of_audio(noise_suppressor: NoiseSuppressor, source_file, _dest_file) -> Statistics:
    raw_y, sr = librosa.load(source_file)

    y = noise_suppressor.just_crop_ends(raw_y, sr)
    if len(y) <= sr * 1:
        raise Exception('Length of audio is too small to be analyzed')

    is_noise, _ = noise_suppressor.noise_sel(y, sr)
    ynoise = y[is_noise]
    signal_sizes = count_sizes(is_noise)

    return Statistics(
        filename = source_file,
        noise_ratio = len(ynoise) / len(y),
        noise_voice_ratio = len(ynoise) / (len(y) - len(ynoise)),
        amount_of_skips = len(signal_sizes) - 1,
        signal_length_avg = np.average(signal_sizes) / sr,
        signal_length_stddev = np.std(signal_sizes) / sr,
    )

def main(argv):
    import csv
    from sys import stdout
    from multiprocessing import cpu_count
    from concurrent.futures import ProcessPoolExecutor

    if len(argv) < 2:
        print(f'usage: {argv[0]} <file_or_folder_to_analyze> [ <file_or_folder_to_analyze> ... ]')
        print(f'generates useful statistic on each file ')
        return -1

    writer = csv.DictWriter(stdout, fieldnames=[field.name for field in fields(Statistics)])
    writer.writeheader()

    def completed_action(file_path: str, future: Future):
        if future.exception() is not None:
            print(f'error processing file {file_path}: {future.exception()}', file=sys.stderr)
            return
        writer.writerow(asdict(future.result()))

    noise_suppressor = NoiseSuppressor(noise_suppress=False)

    futures = process_directory_raw(argv[1:], None, partial(generate_statistics_of_audio, noise_suppressor), completed_action, ['.DS_Store', '.asd'])
    wait(futures)

    return 0

if __name__ == '__main__':
    from sys import argv, exit
    exit(main(argv))