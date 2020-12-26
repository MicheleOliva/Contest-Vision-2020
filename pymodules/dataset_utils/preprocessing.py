import numpy as np
from data_sample import AgeEstimationSample

VGGFACE2_MEANS = np.array([91.4953, 103.8827, 131.0912])

class CustomPreprocessor():
    def __init__(self):
        pass
    
    def _mean_std_normalize(self, inp, means):
        assert(len(inp.shape)>=3)
        d = inp.shape[2]
        outim = np.zeros(inp.shape)
        for i in range(d):
            outim[:,:,i] = (inp[:,:,i] - means[i])
        return outim
    
    def _augment_roi(self, roi, aug_factor):
        aug_roi = []
        w = roi[2]
        h = roi[3]
        new_h = roi[3]*aug_factor
        new_w = roi[2]*aug_factor
        new_x = roi[0] - int((new_w - roi[2])/2)
        new_y = roi[1] - int((new_h - roi[3])/2)
        aug_roi.append(np.max([0, new_x]))
        aug_roi.append(np.max([0, new_y]))
        aug_roi.append(int(new_w))
        aug_roi.append(int(new_h))
        return aug_roi        

    def cut(self, img, roi):
        real_img = img
        aug_roi = self._augment_roi(roi, 1.3)
        img = img[aug_roi[1]:np.min([aug_roi[1]+aug_roi[3], img.shape[0]]),
                  aug_roi[0]:np.min([aug_roi[0]+aug_roi[2], img.shape[1]])]
        if img.size == 0:
            return real_img
        return img
    
    def post_augmentation(self, data):
        return data + VGGFACE2_MEANS


    # data is an array (batch)
    def pre_augmentation(self, data):
        processing = []
        for sample in data:
            roi_list = [sample.roi['upper_left_x'], sample.roi['upper_left_y'], sample.roi['width'], sample.roi['height']]
            img = self.cut(sample.img, roi_list)
            new_roi = {
                'upper_left_x': 0,
                'upper_left_y': 0,
                'width': img.shape[1],
                'height': img.shape[0]
            }
            processing.append(AgeEstimationSample(img, new_roi, sample.age, 'BGR'))
        return processing

    def post_augmentation(self, data):
        processing = []
        for sample in data:
            processing.append(AgeEstimationSample(self.post_augmentation(sample), sample.roi, sample.age, 'BGR'))
        return processing
