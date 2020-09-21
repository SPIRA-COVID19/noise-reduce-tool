import numpy as np
from dataclasses import dataclass, fields, asdict
from pathlib import Path
import librosa

import main as cli_main

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

def generate_statistics_of_audio(filename, source_file) -> Statistics:
    raw_y, sr = cli_main.load_file(source_file)

    y = cli_main.just_crop_ends(raw_y, sr)
    if len(y) <= sr * 1:
        return None

    is_noise, _ = cli_main.noise_sel(y, sr)
    ynoise = y[is_noise]
    signal_sizes = count_sizes(is_noise)

    return Statistics(
        filename = filename,
        noise_ratio = len(ynoise) / len(y),
        noise_voice_ratio = len(ynoise) / (len(y) - len(ynoise)),
        amount_of_skips = len(signal_sizes) - 1,
        signal_length_avg = np.average(signal_sizes) / sr,
        signal_length_stddev = np.std(signal_sizes) / sr,
    )

def path_iterator(paths):
    IGNORED_PATHS = ['.DS_Store', '.asd']

    for search_path in paths:
        if any(x in str(search_path) for x in IGNORED_PATHS):
            continue

        if Path(search_path).is_file():
            just_name = str(Path(search_path).relative_to(Path(search_path).parent)).lstrip('./')
            yield search_path, just_name
            continue

        for path in Path(search_path).rglob('*'):
            generator = path_iterator([path])
            if generator is not None:
                yield from generator

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

    def completed_action(future):
        if future.result() is not None:
            writer.writerow(asdict(future.result()))

    with ProcessPoolExecutor(max_workers=2*cpu_count()) as pool:
        for source_path, filename in path_iterator(argv[1:]):
            future = pool.submit(generate_statistics_of_audio, filename, source_path)
            future.add_done_callback(completed_action)

    return 0

if __name__ == '__main__':
    from sys import argv, exit
    exit(main(argv))