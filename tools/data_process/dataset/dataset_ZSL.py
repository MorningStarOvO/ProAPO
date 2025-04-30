# ==================== Import Packages ==================== #
import time
import sys
import os 

import numpy as np 
import json 

import torch 

from PIL import Image
from torch.utils.data import Dataset


# ==================== Functions ==================== #
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB') 
    

class dataset_ZSL(Dataset):
    def __init__(self, data_root, data_dict, label_to_category_name, transform=None):
        super().__init__()

        self.transform = transform

        category_name_to_label = {}
        for temp_label in label_to_category_name:
            category_name_to_label[label_to_category_name[temp_label]] = int(temp_label)

        self.data_root = data_root
        self.path_image_list = []
        self.label_list = []
        for temp_path in data_dict:
            
            # for Few-Shot data
            if " ***" in temp_path:
                temp_path = temp_path.split(" ***")[0]

            self.path_image_list.append(temp_path)
            temp_label = category_name_to_label[data_dict[temp_path]]
            self.label_list.append(temp_label)


    def __getitem__(self, index):
        path_image = os.path.join(self.data_root, self.path_image_list[index])
        label = self.label_list[index]
        
        img = pil_loader(path_image)
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label)

        return img, label

    def __len__(self):
        return len(self.path_image_list)
    
