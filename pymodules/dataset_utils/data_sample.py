# Data sample: contains image, ROI and age (label)
class AgeEstimationSample():
    def __init__(self, img, roi, label):
        if img is None or label is None:
            raise TypeError()

# Test sample: contains image and ROI 
class FaceSample():
    pass