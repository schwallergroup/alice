"Class for provided numerical descriptors"
from alice.featurization.base import Featurizer

class NumericFeaturizer(Featurizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

