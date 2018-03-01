import cv2
import numpy as np
from PIL import Image

# channel standard deviations
STD = np.array([70.53946096, 51.71475228, 43.03428563], dtype=np.float32)

# channel means
MEAN = np.array([108.64628601, 75.86886597, 54.34005737], dtype=np.float32)

def histogram_equalize(img):
    '''img: RGB numpy image (HxWxC)'''
    # convert the RGB image to YUV format
    img_yuv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output

def normalize(img, mean=MEAN, std=STD):
    '''img: RGB numpy image (HxWxC)
       mean: RGB channels mean
       std: RGB channels std'''
    img_output = np.copy(img)
    for i, (m, s) in enumerate(zip(mean, std)):
        img_output[:,:,i] = (img[:,:,i] - m) / s
    return img_output

def compute_mean(files_imgs, batch_size=128):
    '''files_imgs: files or images to compute mean'''
    m = np.zeros(3)
    shape = None
    for i in range(0, len(files_imgs), batch_size):
        print("done with {:>3} / {} images".format(i, len(files_imgs)))
        images = load_image(files_imgs[i : i + batch_size])
        shape = images.shape
        m += images.sum(axis=(0, 1, 2))
    n = len(files_imgs) * shape[1] * shape[2]
    return (m / n).astype(np.float32)

def compute_std(files_imgs, batch_size=128):
    '''files_imgs: files or images to compute std'''
    s = np.zeros(3)
    s2 = np.zeros(3)
    shape = None
    for i in range(0, len(files_imgs), batch_size):
        print("done with {:>3} / {} images".format(i, len(files_imgs)))
        images = load_image(files_imgs[i : i + batch_size])
        shape = images.shape
        s += images.sum(axis=(0, 1, 2))
        s2 += np.power(images, 2).sum(axis=(0, 1, 2))
    n = len(files_imgs) * shape[1] * shape[2]
    var = (s2 - s**2.0 / n) / (n - 1)
    return np.sqrt(var)

def load_image(fname):
    '''return image as numpy array (HxWxC)
              images as numpy array (NxHxWxC)'''
    if isinstance(fname, str):
        return np.array(Image.open(fname), dtype=np.float32)
    elif isinstance(fname, np.ndarray):
        return fname.astype(np.float32)
    else:
        return np.array([load_image(f) for f in fname])