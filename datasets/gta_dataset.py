import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import sys
import os
import glob
from PIL import Image


class GTA(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None, augmentation=None):
        super(GTA, self).__init__()
        self.image_dir = image_dir # images
        self.mask_dir = mask_dir # gtFine
        self.transform = transform
        self.images = []

        # Iterate over images of each city
        for filename in os.listdir(self.image_dir):
            self.images.append(os.path.join(self.image_dir, filename))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Paths to images and masks
        img_path = self.images[index]
        mask_path = self.images[index].replace('images', 'labels')
        
        # load the images
        image = cv2.imread(img_path)
        
        # Load the masks
        mask = cv2.imread(mask_path)

        mask = self.transform_mask(mask)
        
        if self.transform is not None:
            img = self.transform(image=image, mask=mask)
            t_image = img['image']
            t_mask = img['mask']
            
        # Transform the images into tensors and normalize.
        t = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                               ])
            
        t_image = t(t_image)

        t_mask = torch.from_numpy(t_mask).long()
        
        return t_image, t_mask
    
    def transform_mask(self,mask):
        mapping = {
            (128, 64, 128) : 0,
            (244, 35, 232) : 1,
            (70, 70, 70) : 2,
            (102, 102, 156) : 3,
            (190, 153, 153) : 4,
            (153, 153, 153) : 5,
            (250, 170, 30) : 6,
            (220, 220, 0) : 7,
            (107, 142, 35) : 8,
            (152, 251, 152) : 9,
            (70, 130, 180) : 10,
            (220, 20, 60) : 11,
            (255, 0, 0) : 12,
            (0, 0, 142) : 13,
            (0, 0, 70) : 14,
            (0, 60, 100) : 15,
            (0, 80, 100) : 16,
            (0, 0, 230) : 17,
            (119, 11, 32) : 18
        }
        
        max_color_value = 255
        lookup = np.full((max_color_value + 1, max_color_value + 1, max_color_value + 1), 255, dtype=np.uint8)
        
        for color, label in mapping.items():
            lookup[color] = label

        transformed_mask = lookup[mask[..., 2], mask[..., 1], mask[..., 0]]

        return transformed_mask