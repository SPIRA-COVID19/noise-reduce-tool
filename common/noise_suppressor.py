import numpy as np
import soundfile as sf
import librosa

from .noisereduce import reduce_noise
from .textgrid_writer import audio_to_textgrid, write_textgrid_to_file


class NoiseSuppressor:

    __DEFAULTS = {
        'noise_threshold_db': None,
        'noise_threshold_pct': 0.34,
        'bool_filter_window_size': None,
        'std_threshold': 1.5,
        'suppresion_pct': 1.0,
        'noise_suppress': True,
        'generate_textgrid': False
    }
    
    def __init__(self, **kwargs):
        """
            Creates a Noise Supressor, able to both cut parts of audio
            deemed as preliminary noise, and to create a spectral analysis
            of that preliminary noise, reducing the noise in the whole audio.

            It can receive the following parameters, with each respective default in parentheses:
            noise_threshold_db (None): 
                Define the threshold of dB where above it we consider as signal
                and below we consider as noise. Ranges from -Inf to 0, or None
                if we should consider noise_threshold_pct instead.

            noise_threshold_pct (0.27):
                Instead of defining a dB threshold, get a percentage of the range between the
                minimum dB and maximum dB observed in the signal. Ranges from 0 to 1.
                If noise_threshold_db is defined, it is used instead.

            bool_filter_window_size (0.2 * sample_rate_of_audio):
                We use a boolean majority filter to ease the process of declaring pieces 
                of audio as preliminary noise or not. This decides how big the window is.
                The default considers only noise sections bigger than 0.2 seconds in the
                audio. Ranges from 1 to the amounts of sample in the audio (sample rate * seconds).

            std_threshold (1.5):
                After construction of the mean spectrum of the noise, this parameter
                dictates how far from the mean, in sigmas, we need to be to consider
                the frequency as signal. Ranges from 0 to Infinity.

            suppression_pct (1.0):
                How much of the final audio should be supressed, versus how much of the audio
                should just remain the dry signal from the input. Ranges from 0 to 1, where
                0 returns the original audio and 1 returns just the suppressed audio.


            Alongside these parameters, you have the following options when processing an audio:

            noise_suppress (True):
                Choose whether to suppress noise or to just cut preliminary noise from the
                beginning and ending of an audio.

            generate_textgrid (False):
                Generate a Praat textgrid containing sections where we detected we have signal/noise.
        """
        self.__dict__ = { **self.__DEFAULTS, **kwargs }

    def noise_reduce_signal(self, y, sr):
        # We can only work with audios longer than 1 second, because
        # we will throw away at least 0.5s off of each side
        if len(y) <= sr * 1:
            return y, np.zeros(len(y))

        inoise, _ = self.noise_sel(y, sr)
        noise = y[inoise]

        reduced_y, ε = reduce_noise(audio_clip=y,
                                    noise_clip=noise,
                                    n_grad_freq=4,
                                    n_grad_time=8,
                                    n_std_thresh=self.std_threshold,
                                    prop_decrease=self.suppresion_pct,
                                    verbose=False)

        reduced_y = self.__cut_noise_from_edges(reduced_y, inoise)

        # Normalize to [-1, 1]
        reduced_y /= max(max(y), -min(y), 1)
        ε /= max(max(ε), -min(ε), 1)

        return reduced_y, ε

    def just_crop_ends(self, y, sr):
        if len(y) <= sr * 1:
            return y

        inoise, _ = self.noise_sel(y, sr)
        return self.__cut_noise_from_edges(y, inoise)

    def process_signal_file(self, filename, save_to):
        y, sr = librosa.load(filename, sr=44100)
        y = self.__remove_dc(y)
        if self.noise_suppress:
            reduced_y, _ = self.noise_reduce_signal(y, sr)
        else:
            reduced_y = self.just_crop_ends(y, sr)

        if self.generate_textgrid:
            isnoise, _ = self.noise_sel(reduced_y, sr)
            inoise = np.where(isnoise == True)[0]
            tg = audio_to_textgrid(reduced_y, sr, inoise)
            write_textgrid_to_file(f'{save_to}.TextGrid', save_to, tg)

        sf.write(save_to, reduced_y, sr)
        return filename

    def __remove_dc(self, y):
        """
            Remove any DC from audio, centralizing it at 0 on the range of [-1, 1].
        """
        return y[:] - np.mean(y)

    def __sliding_window_energy(self, y, sr, window_size=4096):
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

    def __boolean_majority_filter(self, y, window_size):
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
    
    def __cut_noise_from_edges(self, y, is_noise):
        """
            Cuts all the noise from the beginning and end of the signal.
            this is made using the indices of inoise.
        """
        # noise = True, signal = False. It returns rows and columns, we just want the rows.
        isignal, *_ = np.where(is_noise == False)
        first_signal, last_signal = isignal[0], isignal[-1]

        return y[first_signal:last_signal]
    
    def noise_sel(self, y, sr, noise_threshold: float = None, eliminate_noise_bigger_than_seconds: float = 0.2):
        edB, edBmin, edBmax = self.__sliding_window_energy(y, sr)

        noise_threshold = self.noise_threshold_db
        if noise_threshold is None:
            noise_threshold = self.noise_threshold_pct * (edBmax - edBmin)

        # select frames with RMS mean next to the minimum level
        is_noise_pre = edB < edBmin + noise_threshold

        window_size = self.bool_filter_window_size or 0.2 * sr

        is_noise = self.__boolean_majority_filter(is_noise_pre, int(window_size))

        return is_noise, is_noise_pre