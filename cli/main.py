from pathlib import Path
from os import makedirs

from common.noise_supressor import NoiseSupressor

def path_iterator(paths, output_path):
    IGNORED_PATHS = ['.DS_Store', '.asd']

    for search_path in paths:
        if any(x in str(search_path) for x in IGNORED_PATHS):
            continue

        if Path(search_path).is_file():
            just_name = str(Path(search_path).relative_to(
                Path(search_path).parent)).split('.')[0]
            makedirs(Path(output_path) /
                     Path(search_path).relative_to(search_path).parent, exist_ok=True)
            yield search_path, f'{output_path}/{just_name}.cleaned.wav'
            continue

        for path in Path(search_path).rglob('*'):
            generator = path_iterator([path], f'{output_path}/{path.relative_to(search_path).parent}')
            if generator is not None:
                yield from generator

def main():
    from multiprocessing import cpu_count
    from concurrent.futures import ProcessPoolExecutor
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='A tool to reduce noise, clip, and mark Praat\'s TextGrids of voice audios.\n' +
                    'This tool is/was used for the SPIRA project.',
        usage='%(prog)s [options] DEST_DIR SOURCE_DIR [SOURCE_DIR ...]',
    )
    parser.set_defaults(noise_suppress=False, generate_textgrid=False, workers=2 * cpu_count())

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    parser.add_argument('--noise-suppress', help='activates noise suppression for the audio processing', action='store_true')
    parser.add_argument('--generate-textgrid', help='generate a noise-signal textgrid for each audio', action='store_true')
    parser.add_argument('--workers', help='parallelize up to max amount of workers', type=int)

    parser.add_argument('dest_dir', help='directory to save all processed audio')
    parser.add_argument('source_dir', help='directories to search for audios to process', nargs='+')

    args = parser.parse_args()

    output_path = args.dest_dir.rstrip('/')
    makedirs(output_path, exist_ok=True)

    noiseprocessor = NoiseSupressor(noise_suppress=args.noise_suppress, generate_textgrid=args.generate_textgrid)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for source_path, dest_path in path_iterator(args.source_dir, output_path):
            bound_source_path = source_path
            future = pool.submit(noiseprocessor.process_signal_file, bound_source_path, dest_path)
            future.add_done_callback(lambda f: print(f'processed {f.result()}') if f.exception() is None else print(f'error processing {bound_source_path}, exception={f.exception()}'))
    return 0

if __name__ == '__main__':
    from sys import exit
    exit(main())
