class CustomPreprocessor():
    def __init__(self):
        pass

    def _augment_roi(self, roi, aug_factor):
        print(roi)
        aug_roi = []
        w = roi[2]
        h = roi[3]
        new_h = roi[3]*aug_factor
        new_w = roi[2]*aug_factor
        aug_roi.append(roi[0] - int((new_w - roi[2])/2))
        aug_roi.append(roi[1] - int((new_h - roi[3])/2))
        aug_roi.append(int(new_w))
        aug_roi.append(int(new_h))
        print(aug_roi)
        return aug_roi        

    def pre_augmentation(self, img, roi):
        aug_roi = self._augment_roi(roi, 1.3)
        img = img[aug_roi[1]:aug_roi[1]+aug_roi[3], aug_roi[0]:aug_roi[0]+aug_roi[2]]
        return img
    
    def post_augmentation(self, data):
        pass

    # data is an array (batch)
    def apply_preprocessing(self, data):
        pass