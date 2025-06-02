import numpy as np


class MandelBrot():
    def __init__(self, c: np.complex128, max_iter: int = 1000, safe: bool = False):
        self.c = c
        self.max_iter = max_iter
        self.safe = safe
        
    def __iter__(self):
        z = 0 + 0j
        for i in range(self.max_iter):
            z = z * z + self.c
            if self.safe and abs(z) > 2:
                break
            yield z
        	