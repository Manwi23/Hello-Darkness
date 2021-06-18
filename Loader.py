import torch
import rawpy
import numpy as np
import os
from net_utils import get_train_names
from image_preprocess_for_net import pack_raw, preprocess, get_amplification_ratio


class Loader:
    def __init__(self):
        t = max(map(int, get_train_names())) + 1
        self.loaded_img = {
            'long' : [None] * 6000, 
            'short' : {
                100 : [None] * t,
                250 : [None] * t,
                300 : [None] * t
            }
        }
        self.short_ex_img_path = './dataset/Sony/short/'
        self.long_ex_img_path = './dataset/Sony/long/'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.short_ex_img = os.listdir(self.short_ex_img_path)
        self.long_ex_img = os.listdir(self.long_ex_img_path)

    def make_batch(self, n, names_list):
        short_img_batch = torch.zeros(n, 4, 512, 512, dtype=torch.float32, device=self.device)
        long_img_batch = torch.zeros(n, 3, 1024, 1024, dtype=torch.float32, device=self.device)
        
        for i, name in enumerate(names_list):
            s_images  = [img for img in self.short_ex_img if img.startswith(name) and img.endswith('.ARW')]
            image = s_images[np.random.randint(0, len(s_images))]
            s_path = self.short_ex_img_path + image
            l_path = self.long_ex_img_path + [img for img in self.long_ex_img if img.startswith(name) and img.endswith('.ARW')][0]

            amp_ratio = get_amplification_ratio(s_path, l_path)
            s_img, l_img = self.load_image(s_path, l_path, name, amp_ratio)
            s_img, l_img = preprocess(s_img, l_img, amp_ratio)
            short_img_batch[i] = s_img
            long_img_batch[i] = l_img
        return short_img_batch, long_img_batch

    def load_image(self, s_path, l_path, name, amp_ratio):
        
        if self.loaded_img['long'][int(name)] is None:
            long_raw = rawpy.imread(l_path)
            long_img = long_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            long_img = np.float32(long_img / 65535.0)
            long_img = torch.tensor(long_img)
            long_img = torch.clamp(long_img, min=0.0, max=1.0)
            self.loaded_img['long'][int(name)] = long_img
        else:
            long_img = self.loaded_img['long'][int(name)]
            
        
        if self.loaded_img['short'][amp_ratio][int(name)] is None:
            short_raw = rawpy.imread(s_path)
            short_img = pack_raw(short_raw)
            short_img = short_img * amp_ratio
            short_img = torch.clamp(short_img, min=0.0, max=1.0)
            self.loaded_img['short'][amp_ratio][int(name)] = short_img
        else:
            short_img = self.loaded_img['short'][amp_ratio][int(name)]
            
        return short_img, long_img