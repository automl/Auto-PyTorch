from autoPyTorch.components.preprocessing.resampling_base import ResamplingMethodBase

class RandomOverSamplingWithReplacement(ResamplingMethodBase):
    def resample(self, X, y, target_size_strategy):
        from imblearn.over_sampling import RandomOverSampler as imblearn_RandomOverSampler
        resampler = imblearn_RandomOverSampler(sampling_strategy=target_size_strategy)
        return resampler.fit_resample(X, y)


class RandomUnderSamplingWithReplacement(ResamplingMethodBase):
    def resample(self, X, y, target_size_strategy):
        from imblearn.under_sampling import RandomUnderSampler as imblearn_RandomUnderSampler
        resampler = imblearn_RandomUnderSampler(sampling_strategy=target_size_strategy)
        return resampler.fit_resample(X, y)