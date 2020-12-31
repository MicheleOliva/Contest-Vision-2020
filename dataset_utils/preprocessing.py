import numpy as np
import cv2
from dataset_utils.data_sample import AgeEstimationSample

VGGFACE2_MEANS = np.array([91.4953, 103.8827, 131.0912])

class CustomPreprocessor():
    def __init__(self, desired_shape=(224, 224)):
        self.desired_shape = desired_shape
    
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
    
    def _subtract_vgg_means(self, data):
        return data - VGGFACE2_MEANS
    
    def _resize(self, sample):
        # Ci prendiamo le dimensioni attuali e quelle desiderate
        h, w = sample.shape[:2]
        dh, dw = self.desired_shape
        sample = sample.astype(dtype=np.uint8)
        
        # Scegliamo un metodo di interpolazione
        if h > dh or w > dw: # Downscaling
            interpolation = cv2.INTER_AREA
        else: # Upscaling
            interpolation = cv2.INTER_CUBIC
        
        # Creiamo un quadrato nero in cui applichiamo l'immagine
        max_dim = max(h, w)
        padded = np.zeros(shape=(max_dim, max_dim, 3), dtype=np.uint8)
        offset_h = int((max_dim - h)/2)
        offset_w = int((max_dim - w)/2)
        padded[offset_h:offset_h + h, offset_w:offset_w + w, :] = sample

        # Restituiamo il resize dell'immagine paddata
        return cv2.resize(padded, dsize=self.desired_shape, interpolation=interpolation).astype(np.float32)


    def _bgr_to_rgb(self, sample):
        return np.flip(sample, 2)


    # data is an array (batch)
    def pre_augmentation(self, data, rois):
        processing = []
        for sample, roi in zip(data, rois):
            roi_list = [roi['upper_left_x'], roi['upper_left_y'], roi['width'], roi['height']]
            processing.append(self.cut(sample, roi_list))
        return processing

    def post_augmentation(self, data):
        processing = []
        for sample in data:
            sample = self._resize(sample)
            sample = self._subtract_vgg_means(sample)
            sample = self._bgr_to_rgb(sample)
            processing.append(sample)
        return processing
