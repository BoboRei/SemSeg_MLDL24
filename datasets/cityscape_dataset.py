import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import sys
import os
import glob
from PIL import Image


class CityScapes(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super(CityScapes, self).__init__()
        self.image_dir = image_dir # images
        self.mask_dir = mask_dir # gtFine
        self.transform = transform
        self.images = []

        # Save all the train images paths
        for city_dir in os.listdir(image_dir):
            city_path = os.path.join(image_dir, city_dir)
            # Check if is a directory
            if os.path.isdir(city_path):
                # Iterate over images of each city
                for filename in os.listdir(city_path):
                    self.images.append(os.path.join(city_path, filename))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Paths to images and masks
        img_path = self.images[index]
        mask_path = self.images[index].replace('images', 'gtFine').replace('leftImg8bit', 'gtFine_labelTrainIds')
        
        # load the images
        image = cv2.imread(img_path)

        # Load the masks
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            img = self.transform(image=image)
            msk = self.transform(image=mask)
            t_image = img['image']
            t_mask = msk['image']
            
        t = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                               ])
        t_image = t(t_image)

        t_mask = torch.from_numpy(t_mask).long()
        
        return t_image, t_mask