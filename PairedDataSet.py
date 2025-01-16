'''
Author: Robber swag162534@outlook.com
Date: 2024-12-08 23:46:13
LastEditors: Robber swag162534@outlook.com
LastEditTime: 2024-12-10 19:47:50
FilePath: \research\su\kneecodes\Region-Attention-Transformer-for-Medical-Image-Restoration\PairedDataSet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from utils import load_img, save_img, load_itk, save_itk, tensor2img, img2tensor 
import torch 
from torch.utils.data import Dataset 
import os 
import glob 
import random



class PairedData(Dataset):
    def __init__(self, root, mask_root, target = 'train', use_fine_mask = True, use_coarse_mask=False, use_num=-100):
        super(Dataset, self).__init__()
        self.use_fine_mask = use_fine_mask
        self.use_coarse_mask = use_coarse_mask

        name_list = os.listdir(os.path.join(root, target))
        name_list_mask = os.listdir(os.path.join(mask_root, target))

        self.HR_path = []
        self.LR_path = []
        self.MASK_fine = []
        self.MASK_coarse = []
        for i, name in enumerate(name_list):
            self.HR_path.append(os.path.join(root, target, name))
            self.LR_path.append(os.path.join(root, target, name))
            if i==use_num-1:
                break
        for i, name in enumerate(name_list_mask):
            self.MASK_fine.append(os.path.join(mask_root, target,name))
            self.MASK_coarse.append(os.path.join(mask_root, target,name))
            if i==use_num-1:
                break

        self.length = len(self.HR_path)
        self.target = target


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        hr_img = img2tensor(load_img(self.HR_path[idx], grayscale=True))
        lr_img = img2tensor(load_img(self.LR_path[idx], grayscale=True))
        # print(hr_img.shape)
        # print(lr_img.shape)

        hr_img = self.resize_image(hr_img)
        lr_img = self.resize_image(lr_img)

        # print(hr_img.shape)
        # print(lr_img.shape)

        _, file_name = os.path.split(self.LR_path[idx])

        if self.use_fine_mask:
            mask_fine = img2tensor(load_img(self.MASK_fine[idx], grayscale=True))
        else:
            mask_fine = torch.zeros_like(hr_img)

        if self.use_coarse_mask:
            mask_coarse = img2tensor(load_img(self.MASK_coarse[idx], grayscale=True))
        else:
            mask_coarse = torch.zeros_like(hr_img)

        # print(mask_fine.shape)
        # print(mask_coarse.shape)

        mask_fine = self.resize_image(mask_fine)
        mask_coarse = self.resize_image(mask_coarse)

        # print(mask_fine.shape)
        # print(mask_coarse.shape)

        if self.target == "train":
            i = random.choice([1, 2, 3, 4])
            hr_img = torch.rot90(hr_img, i, [1, 2])
            lr_img = torch.rot90(lr_img, i, [1, 2])
            if self.use_fine_mask:
                mask_fine = torch.rot90(mask_fine, i, [1, 2])
            if self.use_coarse_mask:
                mask_coarse = torch.rot90(mask_coarse, i, [1, 2])

        return hr_img, lr_img, mask_fine, mask_coarse, file_name

    def resize_image(self, img):
        target_size = (512, 512)
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
        return img.squeeze(0)



