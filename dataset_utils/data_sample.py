# Data sample: contains image, ROI and age (label)
class AgeEstimationSample():
    def __init__(self, img, roi, label, img_mode):
        if img is None or label is None:
            raise TypeError('Image and label must always be specified')

        self.allowed_img_modes = ['BGR', 'RGB']
        if img_mode not in self.allowed_img_modes:
            error_msg = f"'img_mode' must be one of {self.allowed_img_modes}"
            raise TypeError(error_msg)

        self.img = img
        self.roi = roi
        self.age = label
        self.img_mode = img_mode

# Test sample: contains image and ROI 
class FaceSample():
    pass