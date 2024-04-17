import numpy as np
import torch
import random
import cv2

from mobile_sam.utils.transforms import ResizeLongestSide
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import functional as TF
from pycocotools.coco import COCO
import os

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

    return masks_tensor,torch.tensor(unique_ids)


def random_horizontal_flip(image, mask, p=0.5):
    if random.random() > p:
        return TF.hflip(image), TF.hflip(mask)
    return image, mask

def random_rotation(image, mask, degrees):
    angle = random.uniform(-degrees, degrees)
    return TF.rotate(image, angle), TF.rotate(mask, angle)

def apply_transforms_both(image, mask, p_flip=0.5, degrees=0):
    image, mask = random_horizontal_flip(image, mask, p=p_flip)
    image, mask = random_rotation(image, mask, degrees=degrees)
    return image, mask


class SamDataset(Dataset):
    def __init__(self, images_dir, masks_dir, imgs_num=None, image_specific_transform=None,
                 mask_and_image_transform=True,p_flip =0.5,degrees=90):
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
        self.p_flip = p_flip
        self.degrees = degrees
        self.mask_and_image_transform = mask_and_image_transform
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
            image, mask = apply_transforms_both(image, Image.fromarray(mask),self.p_flip,self.degrees)
        if self.image_specific_transform is not None:
            image = self.image_specific_transform(image)
        masks,unique_ids = process_mask_image(np.array(mask))

        sample = {'image': image, 'mask': masks,'unique_ids':unique_ids}

        return sample


class COCODataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.coco = COCO(annotation)
        self.transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.supercategory_to_id = {}
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_to_supercategory = {cat['id']: cat['supercategory'] for cat in self.categories}
        self.supercategory_to_id = {}
        categories = self.coco.loadCats(self.coco.getCatIds())
        for cat in categories:
            supercat = cat['supercategory']
            if supercat not in self.supercategory_to_id:
                self.supercategory_to_id[supercat] = len(self.supercategory_to_id)

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = read_image(os.path.join(self.root, path))

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Create a dictionary to combine masks by category ID
        combined_masks = {}
        for ann in anns:
            category_id = ann['category_id']
            super_catagory = self.category_to_supercategory[category_id]
            category_id = self.supercategory_to_id[super_catagory]
            mask = self.coco.annToMask(ann)
            if category_id in combined_masks:
                # Combine masks by logical OR
                combined_masks[category_id] = np.logical_or(combined_masks[category_id], mask).astype(np.uint8)
            else:
                combined_masks[category_id] = mask

        # Convert combined masks to tensors

        masks = [torch.tensor(combined_masks[key], dtype=torch.uint8) for key in sorted(combined_masks)]
        category_ids = torch.tensor(list(sorted(combined_masks.keys())), dtype=torch.int64)

        if self.transforms:
            image, masks = self.transforms(image, masks)
        if len(masks) == 0:
            sample = {'image': image, 'mask': torch.zeros((1, image.shape[1], image.shape[2])),
                      'unique_ids': torch.zeros(1)}
            return sample
        masks = torch.stack(masks)
        sample = {'image': image, 'mask': masks, 'unique_ids': category_ids}

        return sample

    def __len__(self):
        return len(self.ids)

    def transforms(self, image, masks):
        image = TF.convert_image_dtype(image, dtype=torch.float)
        masks = [torch.tensor(mask, dtype=torch.uint8) for mask in masks]
        return image, masks

    def get_category_names(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        category_names = {cat['id']: cat['name'] for cat in categories}
        return category_names