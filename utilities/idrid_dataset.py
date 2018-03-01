import imghdr
import os
import numpy as np

from . import loader_base


class idrid_dataset_creator():

    def __init__(self, raw_folder, label_folder, label_suffix_ext, ids = None, transform = None, normalizer = None, cache = True):
        self.raw_folder = raw_folder
        self.label_folder = label_folder
        self.label_suffix_ext = label_suffix_ext
        self.ids = ids
        self.transform = transform
        self.normalizer = normalizer
        self.cache = cache

    def _read_data_location(self, raw_folder, label_folder):
        """implemntation details of idrid loader.
        This function read the files location of raw images and masks
        """
        raw_names = []    
        label_names = []    
        valid_types = ['jpg', 'jpeg', 'png', 'bmp', 'gif']            
        for root, dirs, files in os.walk(raw_folder):
            for i, f in enumerate(files):
                filename = os.path.splitext(f)[0]
                if (self.ids is not None and int(filename[-2:]) not in self.ids):
                    continue
                label_name = filename + self.label_suffix_ext                
                raw_type = imghdr.what(raw_folder + f)    
                if raw_type in valid_types:
                    #print(i, "is valid type:", "open raw file:", os.path.join(raw_folder, f), 
                    #      "open label file:", os.path.join(label_folder, label_name))
                    raw_names.append(raw_folder + f)
                    label_names.append(label_folder + label_name)
        return raw_names, label_names

    def __call__(self):
        return loader_base.data_loader(self.raw_folder, self.label_folder, self._read_data_location, self.transform, self.normalizer, self.cache)