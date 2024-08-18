from BoradLearningSystem import *

class ConvBLS(BLS):
    def __init__(self, NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2 ** -30, NumBlock=3, is_argmax=False):
        super().__init__()
        self.NumBlock = NumBlock