"""
    Some of the functions here used are taken from https://github.com/MiviaLab/GenderRecognitionFramework .
"""
import numpy as np
from skimage.filters import gaussian
import random
import skimage as sk
import cv2
from scipy import ndimage


def rotation(image, degrees):
    return ndimage.rotate(image, degrees, reshape=False) # no reshape reduces the area that is padded to zero

def gaussian_noise(x, c):
    """Adds gaussian noise to the image x with scale c.
    Good values for c: .08 to .30"""
    x = np.array(x) / 255.
    x += np.random.normal(size=x.shape, scale=c)
    x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    return x


def gaussian_blur(x, c):
    """Applies gaussian blur to image x with variance c.
    Good values for c: 1.0 to 4.0"""
    x = np.array(x) / 255.
    x = gaussian(x, sigma=c, multichannel=True)
    x = np.clip(x, 0, 1) * 255
    return x


def random_crop(img, factor=0.1):
    width = int(img.shape[1]*(1-factor))
    height = int(img.shape[0]*(1-factor))
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def brightness(x, c):
    """Increases or decreases brightness of image x by a factor c.
    To increase, good values of c: 0.1 to 0.5.
    To decrease, good values of c: -0.1 to -0.5."""
    x = np.array(x) / 255.
    
    if len(x.shape)>2 and x.shape[2]>1:
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)
    else:
        x = np.clip(x + c, 0, 1)
    
    x = np.clip(x, 0, 1) * 255
    return x


def contrast(x, c):
    """Increases or decreases contrast of image x by a factor c.
    To increase, good values of c: 1.5 to 5.0.
    To decrease, good values of c: 0.4 to 0.1."""
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    x = np.clip((x - means) * c + means, 0, 1) * 255
    return x


def horizontal_flip(x):
    """Flips image x on the horizontal axis."""
    return np.flip(np.array(x), axis=1)


def _random_normal_crop(n, maxval, positive=False, mean=0):
    gauss = np.random.normal(mean,maxval/2,(n,1)).reshape((n,))
    gauss = np.clip(gauss, mean-maxval, mean+maxval)
    if positive:
        return np.abs(gauss)
    else:
      return gauss


def skew(img):
    """Warps the image img in the 3-dimensional space randomly."""
    s = _random_normal_crop(2, 0.1, positive=True)
    M=np.array( [ [1,s[0],1], [s[1],1,1]] )
    img = img.astype(np.uint8)
    nimg = cv2.warpAffine(img, M, dsize=img.shape[0:2])
    if len(nimg.shape)<3:
        nimg = nimg[:,:,np.newaxis]
    nimg = nimg.astype(np.float32)
    return nimg #.reshape(img.shape)

def spatter(x, severity=1):
    """Adds spatter to the image x with severity 1 to 5."""
    iscolor = len(x.shape)>2 and x.shape[2] > 1
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)
        
        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        if len(x.shape)>2 and x.shape[2] > 1:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        x = np.clip(x + m * color, 0, 1) * 255
        if iscolor: 
            return cv2.cvtColor(x, cv2.COLOR_BGRA2BGR)
        else:
            return cv2.cvtColor(x, cv2.COLOR_BGRA2GRAY)
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        x = np.clip(x + color, 0, 1) * 255
        
        if iscolor: 
            return x
        else:
            return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)