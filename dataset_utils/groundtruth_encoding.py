import numpy as np

class OrdinalRegressionEncoder():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def encode(self, batch):
        return self._real_to_ordinal(batch)

    def _real_to_ordinal(self, batch):
        ret_array = []
        for real_value in batch:
            nearest_integer = round(real_value)
            dtype = np.float16 # must be float, otherwise binary crossentropy can't be computed
            ordinal_gt_lower = np.ones(nearest_integer+1, dtype=dtype) # e.g. for 25 must be [0,25], so 26 ones
            ordinal_gt_upper = np.zeros(self.num_classes-(nearest_integer+1), dtype=dtype)
            ret_array.append(np.concatenate((ordinal_gt_lower, ordinal_gt_upper)))
        return np.array(ret_array)
