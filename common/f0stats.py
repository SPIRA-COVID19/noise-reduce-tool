from .noise_suppressor import NoiseSuppressor
from dataclasses import dataclass
import numpy as np
import librosa

@dataclass
class F0Statistics:
    median: float
    mean: float
    std: float
    min: float
    max: float

# print(np.median(f0final),end=',')
# print(np.mean(f0final),end=',')
# print(np.std(f0final),end=',')
# print(np.min(f0final),end=',')
# print(np.max(f0final))

class F0StatisticsExtractor(NoiseSuppressor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_f0_statistics(self, y, sr) -> F0Statistics:
        inoise, _ = self.noise_sel(y, sr)
        isignal = np.logical_not(inoise)
        signal = y[isignal]

        if len(signal) == 0:
            return F0Statistics(0.0, 0.0, 0.0, 0.0, 0.0)

        f0, pf0, ppf0 = librosa.pyin(signal, sr=sr,fmin=50,fmax=600)
        f0final= f0[~np.isnan(f0)]
        return F0Statistics(
            median=np.median(f0final),
            mean=np.mean(f0final),
            std=np.std(f0final),
            min=np.min(f0final),
            max=np.max(f0final),
        )
