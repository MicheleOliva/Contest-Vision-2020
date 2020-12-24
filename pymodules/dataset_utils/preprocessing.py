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

    def pre_augmentation(self, img, roi):
        aug_roi = self._augment_roi(roi, 1.3)
        img = img[aug_roi[1]:aug_roi[1]+aug_roi[3], aug_roi[0]:aug_roi[0]+aug_roi[2]]
        return img
    
    def post_augmentation(self, data):
        
        vgg2_means = np.array([91.4953, 103.8827, 131.0912]) # BGR

        outimg = self._mean_std_normalize(data, vgg2_means)

        facemax = np.max(outimg)
        facemin = np.min(outimg)
        outimg = (255 * ((outimg - facemin) / (facemax - facemin))).astype(np.uint8)
        
        if (len(outimg.shape)<3 or outimg.shape[2]<3):
            outimg = np.repeat(np.squeeze(outimg)[:,:,None], 3, axis=2)
        
        return outimg


    # data is an array (batch)
    def apply_preprocessing(self, data):
        pass