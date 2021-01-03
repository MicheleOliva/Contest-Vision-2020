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

    # IN REALTA' NON LA USO PERCHE' I CALCOLI LI FACCIO NELLA METRICA CUSTOM
    def decode(self, batch):
        return self._ordinal_to_int(batch)

    def _ordinal_to_int(self, batch):
        int_predictions = []
        for prediction in batch:
            # e.g. the sum of ones in [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            # 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
            # 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            # 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            # 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            # 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            # 0., 0., 0., 0., 0.] is 26, but the array encodes 25, so there's the need to subtract one

            # CORREGGERE: devo convertire il vettore di probabilità che caccia la rete in un vettore di 0 e 1,
            # e l'età è data dall'indice del primo elemento che è zero, meno uno
            int_value = np.sum(prediction)-1
            int_predictions.append(int_value)
        return np.array(int_predictions)