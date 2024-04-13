import numpy as np
import torch
import random
import torchvision.transforms.functional as TF
import cv2
import os
from torch.utils.data import Dataset
from mobile_sam.utils.transforms import ResizeLongestSide
from PIL import Image

def process_mask_image(mask_image):
    unique_ids = np.unique(mask_image)
    unique_ids = unique_ids[unique_ids != 0]
    mask_info_list = []

    for mask_id in unique_ids:
        mask = mask_image == mask_id
        mask_area = np.sum(mask)
        mask_info_list.append((mask_area, mask))

    masks_list = [mask_info[1] for mask_info in mask_info_list]
    if len(masks_list) == 0:
        return None, None
    masks_tensor = np.stack(masks_list, axis=-1)

    masks_tensor = torch.tensor(masks_tensor).permute(2, 0, 1)

    return masks_tensor,unique_ids


def random_horizontal_flip(image, mask, p=0.5):
    if random.random() > p:
        return TF.hflip(image), TF.hflip(mask)
    return image, mask

def random_rotation(image, mask, degrees):
    angle = random.uniform(-degrees, degrees)
    return TF.rotate(image, angle), TF.rotate(mask, angle)

def apply_transforms_both(image, mask, p_flip=0.5, degrees=180):
    image, mask = random_horizontal_flip(image, mask, p=p_flip)
    image, mask = random_rotation(image, mask, degrees=degrees)
    return image, mask


class SamDataset(Dataset):
    def __init__(self, images_dir, masks_dir, imgs_num=None, image_specific_transform=None,
                 mask_and_image_transform=True, bbox_mod_prob=0, bbox_mod_px=40):
        """
        Custom dataset for semantic segmentation.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.sam_transform = ResizeLongestSide(1024)
        self.image_specific_transform = image_specific_transform
        if imgs_num is not None:
            all_images = sorted(os.listdir(images_dir))  # Get and sort all image filenames
            if imgs_num > len(all_images):
                self.images = all_images  # If imgs_num exceeds available images, use all images
            else:
                self.images = random.sample(all_images, imgs_num)  # Randomly select imgs_num images
        else:
            self.images = sorted(os.listdir(images_dir))  # Use all images if imgs_num is None
        self.mask_and_image_transform = mask_and_image_transform
        self.bbox_mod_px = bbox_mod_px
        self.bbox_mod_prob = bbox_mod_prob

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.images[idx])
        mask_name = os.path.join(self.masks_dir, self.images[idx]).replace('jpg','png')
        image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        image = self.sam_transform.apply_image(image)
        image = torch.as_tensor(image)
        image = image.permute(2, 0, 1).contiguous()[None, :, :, :][0]
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST)
        if self.mask_and_image_transform:
            image, mask = apply_transforms_both(image, Image.fromarray(mask))
        if self.image_specific_transform is not None:
            image = self.image_specific_transform(image)
        masks,unique_ids = process_mask_image(np.array(mask))

        sample = {'image': image, 'mask': masks,'unique_ids':unique_ids}

        return sample