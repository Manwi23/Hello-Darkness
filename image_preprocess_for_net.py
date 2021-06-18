import torch
import torchvision
import torchvision.transforms.functional as ttf
import numpy as np
import random
import os

def preprocess(image, ground_truth, amp_ratio):
    s = image.shape
    if len(s) == 3 and s[2] == 4:
        image = image.permute(2, 0, 1)
        ground_truth = ground_truth.permute(2, 0, 1)
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(image, output_size=(512, 512))
    image = ttf.crop(image, i, j, h, w)
    ground_truth = ttf.crop(ground_truth, 2*i, 2*j, h*2, w*2)

    if random.random() > 0.5:
        image = ttf.hflip(image)
        ground_truth = ttf.hflip(ground_truth)

    if random.random() > 0.5:
        image = ttf.vflip(image)
        ground_truth = ttf.vflip(ground_truth)

    return image, ground_truth


def pack_raw(raw):

    # pack Bayer image to 4 channels & subtract black level
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    
    return torch.tensor(out)


def get_amplification_ratio(img_path, gt_path):
    img_base = os.path.basename(img_path)
    gt_base = os.path.basename(gt_path)
    img_time = float(img_base[9:-5])
    gt_time = float(gt_base[9:-5])
    return round(gt_time / img_time, -1)