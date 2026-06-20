import numpy as np
import math

class HeNormal:

    @staticmethod
    def initialize(fanIn: int, fanOut: int):
        sd = math.sqrt(2.0 / fanOut) #NOTE: Numpy uses a half-open range, so the distribution is uniform from [min, max).
        return np.ones((fanIn, fanOut)) * sd #Values close to 0 have higher likelihood.

class HeUniform:

    @staticmethod
    def initialize(fanIn: int, fanOut: int, **kwargs):
        rng = np.random.default_rng()

        #Checking for optional seed parameter
        seed = kwargs.get("seed", None)
        if (isinstance(seed, int) or (seed is None)):
            rng = np.random.default_rng(seed=seed)

        limit = math.sqrt(6.0 / fanIn)
        return rng.uniform(low=-limit, high=limit, size=(fanIn, fanOut))

class XavierNormal:

    @staticmethod
    def initialize(fanIn: int, fanOut: int):
        sd = math.sqrt(2.0 / (fanIn + fanOut))
        return np.ones((fanIn, fanOut)) * sd

class XavierUniform:
    
    @staticmethod
    def initialize(fanIn: int, fanOut: int):
        sd = math.sqrt(6.0 / (fanIn + fanOut))
        return np.ones((fanIn, fanOut)) * sd

INITIALIZERS = {
    "he_normal": HeNormal(),
    "he_uniform": HeUniform(),
    "xavier_normal": XavierNormal(),
    "xavier_uniform": XavierUniform(),
    "": None,
    None: None
}