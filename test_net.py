import torch
from torch import nn
import rawpy
from PIL import Image
from Unet import U_net
from net_utils import get_test_names, get_last_epoch
from image_preprocess_for_net import pack_raw, get_amplification_ratio
import numpy as np
import os

def test_sony(device, model_name=U_net, saved_model_param='models/Sony', loss=nn.MSELoss(reduction='mean'), loss_name='MSE', locator='cpu'):
    model_loss = []
    order = []
    ratio_order = []
    result_dir = './result_Sony/'
    test_names = get_test_names()
    short_ex_img_path = './dataset/Sony/short/'
    long_ex_img_path = './dataset/Sony/long/'
    model = model_name()
    model.load_state_dict(torch.load(saved_model_param, map_location=locator))
    model = model.to(device)
    model.eval()
    for name, param in model.named_parameters():
        if param.requires_grad:
            pass
    short_ex_img = os.listdir(short_ex_img_path)
    long_ex_img = os.listdir(long_ex_img_path)
    with torch.no_grad():
        test_names = ['10003', '10006', '10011', '10016', '10022', '10030', '10032', '10069', '10106', '10191']
        for name in test_names:
            s_images  = [img for img in short_ex_img if img.startswith(name) and img.endswith('.ARW')][:1]
            l_image = [img for img in long_ex_img if img.startswith(name) and img.endswith('.ARW')][0]
            l_path = long_ex_img_path+l_image
            long_raw = rawpy.imread(l_path)
            long_img = long_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            long_img = np.float32(long_img / 65535.0)
            for img in s_images:
                name_num = img[:8]
                s_path = short_ex_img_path+img
                ratio = get_amplification_ratio(s_path, l_path)
                short_raw = rawpy.imread(s_path)
                short = pack_raw(short_raw).permute(2, 0, 1) * ratio
                short_to_model = torch.unsqueeze(short, 0)
                im = short_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                scale_full = np.float32(im / 65535.0)
                scale_full = scale_full * np.mean(long_img) / np.mean(scale_full)
                output = model(short_to_model.to(device))
                net_loss = loss(torch.unsqueeze(torch.tensor(long_img), 0), output.permute(0, 2, 3, 1))
                model_loss.append(net_loss)
                order.append(img)
                ratio_order.append(ratio)
                output = torch.clamp(output, min=0.0, max=1.0)
                output = output.squeeze().cpu().numpy().transpose((1, 2, 0))
                
                Image.fromarray(np.floor(np.clip(output * 255, 0, 256)).astype(np.uint8), mode='RGB').save(
                  result_dir + 'final/%s_%d_%s_%s_out.png' % (name_num, ratio, model.__class__.__name__, loss_name))
                Image.fromarray(np.floor(np.clip(scale_full*255, 0, 256)).astype(np.uint8), mode='RGB').save(
                  result_dir + 'final/%s_%d_%s_%s_scale.png' % (name_num, ratio, model.__class__.__name__, loss_name))
                Image.fromarray(np.floor(np.clip(long_img * 255, 0, 256)).astype(np.uint8), mode='RGB').save(
                  result_dir + 'final/%s_%d_%s_%s_gt.png' % (name_num, ratio,  model.__class__.__name__, loss_name))
    return model_loss, order, ratio_order