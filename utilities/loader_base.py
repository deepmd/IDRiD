import cv2
import PIL as pil
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#from keras.preprocessing.image import load_img
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


'''if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS
        

def load_img(path, grayscale=False, target_size=None, interpolation='nearest'):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img'''

class data_loader(Dataset):
    """This class is a base class to load data of image segmentation and return raw image and label in a dictionary. 
    Args:
        raw_folder(string) : Location of the raw images
        label_folder(string) : Location of the labels
        location_loader(callback) : This class determine how to load the images and labels location, accept two parameters
        which are the location of raw images and location of labels
        trasnformer(class) : Composition of different transformation
        normalizer(class) : If it is not None, this class will normalize the image. If the cache is True too, 
        loader_base will normalize the image one time only. This class is not meant to composed with regular
        trnasformer of PyTorch
    
    Example:    
    transform_func = transforms.Compose([transform_policy.random_crop((320, 480)), transform_policy.flip_horizontal(), transform_policy.to_tensor()])    
    loader = loader_base("/home/computer_vision_dataset/segmentation/camvid/train_raws/", 
                         "/home/computer_vision_dataset/segmentation/camvid/train_labels/", 
                         transform_func, camvid_loader.loader)
    sample = loader[0]
    raw_img = sample['raw']
    label_img = sample['label']
    """
    
    def __init__(self, raw_folder, label_folder, location_loader, trasnformer = None, normalizer = None, cache = True):
        super(data_loader, self).__init__()

        self._mcache = cache        
        self._mtransform = trasnformer
        self._mraw_names, self._mlabel_names = location_loader(raw_folder, label_folder)
        self._mraw_imgs, self._mlabel_imgs = [], []
        self._mnormalizer = normalizer
        if cache:
            for i in range(len(self._mraw_names)):
                #print(i, "is valid type:", "open raw file:", raw_folder + self._mraw_names[i], "open label file:", label_folder + self._mlabel_names[i])
                raw = pil.Image.open(self._mraw_names[i])
                label = pil.Image.open(self._mlabel_names[i]).convert('L') #mode=L grayscale

                #if normalizer:
                #    raw_img = normalizer(raw_img)
                
                self._mraw_imgs.append(raw)
                self._mlabel_imgs.append(label)                    
        
        print(len(self._mraw_names), len(self._mraw_imgs))
        print(len(self._mlabel_names), len(self._mlabel_imgs))                
        
    def __getitem__(self, idx):
        """This function return two data, raw image and label image        
        """
        idx = idx % len(self._mraw_names) 
        if self._mcache:
            raw = self._mraw_imgs[idx]
            label = self._mlabel_imgs[idx]
        else:
            raw = pil.Image.open(self._mraw_names[idx])
            label = pil.Image.open(self._mlabel_names[idx]).convert('L') #mode=L grayscale
            
        if self._mtransform:            
                raw, label = self._mtransform(raw, label)
            
        return {'raw' : raw, 'label' : label}
        
    def __len__(self):
        return len(self._mraw_names)