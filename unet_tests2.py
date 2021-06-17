# !pip install tensorflow==1.14
# !pip install rawpy
# !pip install scipy==1.2.0

# !git clone https://github.com/Manwi23/Hello-Darkness.git
# !git clone https://github.com/cchen156/Learning-to-See-in-the-Dark.git
# %cd Learning-to-See-in-the-Dark/
# !python download_models.py
# !cd dataset && wget https://storage.googleapis.com/isl-datasets/SID/Sony.zip

import math
#import tensorflow.contrib

import matplotlib.pyplot as plt
import rawpy
import os

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
import torchvision
from torch import nn
import torch.optim as optim
from torch.autograd import Variable

import random
from tqdm import tqdm
from PIL import Image

import numpy as np

from torch.cuda.amp import autocast, GradScaler

def unzip_only_names(names):
    cmd = "cd dataset && unzip -l Sony.zip"
    out = os.popen(cmd).read()
    # print(out)

    cmd = "cd dataset && mkdir Sony"
    os.system(cmd)

    cmd = "cd dataset/Sony && mkdir short"
    os.system(cmd)

    cmd = "cd dataset/Sony && mkdir long"
    os.system(cmd)

    # print(in_testset_long)

    for line in out.split('\n'):
        sp = line.split('/')
        if len(sp) > 1:
            ph = sp[-1]
            name = ph[1:5]
            # print(name)
            if name in names:
                path = ''
                if 'short' in line:
                    path = 'Sony/short/' + ph
                else:
                    path = 'Sony/long/' + ph
                cmd = "cd dataset && unzip -p Sony.zip " + path + " > " + path
                os.system(cmd)

# unzip_only_names(['0001', '0003', '0004'])
# !python test_Sony.py 

def plot_k_model_outputs(k=3, fig_x=20, fig_y=10):
    img_list = os.listdir("result_Sony/final")
    img_list.sort()
    model_path = "result_Sony/final/"
    short_img = 'dataset/Sony/short/'
    fig, axes = plt.subplots(nrows=k, ncols=4, figsize=(fig_x, fig_y))
    [axi.set_axis_off() for axi in axes.ravel()]
    axes[0][0].title.set_text('Short exposure')
    axes[0][1].title.set_text('Long exposure')
    axes[0][2].title.set_text('Model output')
    axes[0][3].title.set_text('Scaled output')
    for i in range(k):
        name = img_list[i*3][:-10]
        gt_raw = rawpy.imread(short_img+name+'0.04s.ARW')
        im = gt_raw.postprocess()
        axes[i][0].imshow(im)
        img = plt.imread(model_path+img_list[i*3])
        axes[i][1].imshow(img)
        img = plt.imread(model_path+img_list[i*3+1])
        axes[i][2].imshow(img)
        img = plt.imread(model_path+img_list[i*3+2])
        axes[i][3].imshow(img)

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
    #print(type(image))
    #image = torch.tensor(image)
    #image = torch.unsqueeze(image, 0)
    
    # print(image.shape, ground_truth.shape)
    return image, ground_truth

def pack_raw(raw):

    # pack Bayer image to 4 channels & subtract black level
    im = raw.raw_image_visible.astype(np.float32)
    # print(im.shape)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level
    ## ja się chętnie pewnego dnia dowiem po co jest to dzielenie

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    
    return torch.tensor(out)

# ex_short = 'dataset/Sony/short/'+'00004_00_'+'0.04s.ARW'
# ex_long = 'dataset/Sony/long/'+'00004_00_'+'10s.ARW'
# gt_raw = rawpy.imread(ex_short)
# out = pack_raw(gt_raw)
# #print(gt_raw.sizes)
# im = gt_raw.postprocess()
# plt.imshow(im)

def get_amplification_ratio(img_path, gt_path):
    img_base = os.path.basename(img_path)
    gt_base = os.path.basename(gt_path)
    img_time = float(img_base[9:-5])
    gt_time = float(gt_base[9:-5])
    return round(gt_time / img_time, -1)

# get_amplification_ratio(ex_short, ex_long)

class Conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True))
            #nn.ReLU())
    
    def forward(self, X):
        return self.layer(X)

    def loss(self, Out, Targets):
        return F.cross_entropy(Out, Targets)

class Down_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_layer, self).__init__()
        self.pool_down = nn.MaxPool2d(2)
        self.conv = Conv_layer(in_channels, out_channels)

    def forward(self, X):
        out = self.pool_down(X)
        out = self.conv(out)
        return out

    def loss(self, Out, Targets):
        return F.cross_entropy(Out, Targets)

class Up_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_layer, self).__init__()
        self.pool_up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = Conv_layer(in_channels, out_channels)

    def forward(self, X_d, X_u):
        X_d= self.pool_up(X_d)
        concat_out = torch.cat([X_u, X_d], 1)
        out = self.conv(concat_out)
        return out

    def loss(self, Out, Targets):
        return F.cross_entropy(Out, Targets)


class U_net(nn.Module):
    def __init__(self):
        super(U_net, self).__init__()
        self.down_layer1 = Conv_layer(4, 32)
        self.down_layer2 = Down_layer(32, 64)
        self.down_layer3 = Down_layer(64, 128)
        self.down_layer4 = Down_layer(128, 256)
        self.down_layer5 = Down_layer(256, 512)

        self.up_layer1 = Up_layer(512, 256)
        self.up_layer2 = Up_layer(256, 128)
        self.up_layer3 = Up_layer(128, 64)
        self.up_layer4 = Up_layer(64, 32)
        self.up_layer5 = nn.Conv2d(32, 12, 1)
        self.out_layer = nn.PixelShuffle(2)


    def forward(self, X):
        d1 = self.down_layer1(X)
        d2 = self.down_layer2(d1)
        d3 = self.down_layer3(d2)
        d4 = self.down_layer4(d3)
        d5 = self.down_layer5(d4)

        u1 = self.up_layer1(d5, d4)
        u2 = self.up_layer2(u1, d3)
        u3 = self.up_layer3(u2, d2)
        u4 = self.up_layer4(u3, d1)
        u5 = self.up_layer5(u4)

        out = self.out_layer(u5)
        return out

    def loss(self, Out, Targets):
        return F.cross_entropy(Out, Targets)



class ConvDeconv(nn.Module):
    def __init__(self):
        super(ConvDeconv, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(4, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 12, 3),
            nn.PixelShuffle(2)
        )

        self.mse = nn.MSELoss()

    def forward(self, x):
        #for layer in self.layers:
        #    x = layer(x)
            #print(x.shape)
        return self.layers(x)

    def loss(self, x, y):
        if y.shape != x.shape:
            trans = torchvision.transforms.CenterCrop((x.shape[2], x.shape[3]))
            y = trans(y)
        return self.mse(x,y)
    
model = ConvDeconv()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def get_train_names():
    short_ex_img_path = './dataset/Sony/short/'
    long_ex_img_path = './dataset/Sony/long/'
    all_from_dir = os.listdir(long_ex_img_path)
    train_images  = [x[0:5] for x in all_from_dir if x.startswith('0') and x.endswith('.ARW')]
    return train_images

def get_test_names():
    short_ex_img_path = './dataset/Sony/short/'
    long_ex_img_path = './dataset/Sony/long/'
    all_from_dir = os.listdir(long_ex_img_path)
    train_images  = [x[0:5] for x in all_from_dir if x.startswith('1') and x.endswith('.ARW')]
    return train_images

class Loader:
    def __init__(self):
        # super().__init__()
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
        short_img_batch = torch.zeros(n, 4, 512, 512, dtype=torch.float32, device=device)
        long_img_batch = torch.zeros(n, 3, 1024, 1024, dtype=torch.float32, device=device)
        
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
        
    #ground_truth = torch.tensor(ground_truth)
    #ground_truth = torch.unsqueeze(ground_truth, 0)
        
        
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
            #print(amp_ratio, short_img.min(), short_img.mean(), short_img.max())
            short_img = short_img * amp_ratio
            #print(amp_ratio, short_img.min(), short_img.mean(), short_img.max())
            short_img = torch.clamp(short_img, min=0.0, max=1.0)
            self.loaded_img['short'][amp_ratio][int(name)] = short_img
        else:
            short_img = self.loaded_img['short'][amp_ratio][int(name)]
            
        return short_img, long_img

try:
    os.makedirs('models/Sony5')
except Exception:
    pass
saved_model_param = 'models/Sony5'

loader = Loader()

def get_last_epoch():
    saved_model_param = 'models/Sony5'

    def get_epoch(path):
        #print(path.split('.')[0].split('-')[-1])
        if "epoch" in path:
            return int(path.split('.')[0].split('-')[-1])
        else:
            return -1

    try:
        last_epoch = max(map(get_epoch, os.listdir(saved_model_param)))
    except:
        return False, 0, None

    return True, last_epoch, '/epoch-%d.pth' % last_epoch

def train_sony(model, train_names, num_epoch, batch_size=1, save_every=10, model_name=U_net):
    learning_rate = 1e-4
    loss = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    model = model.train()
    
    scaler = GradScaler()
    
    for epoch in range(0, num_epoch):
        if epoch > 2000:
            for param in optimizer.param_groups:
                param['lr'] = 1e-5
        #print(epoch)
        order = np.random.permutation(train_names)
        #short_ex_img = os.listdir(short_ex_img_path)
        #long_ex_img = os.listdir(long_ex_img_path)
        num_img = len(order)
        n = math.ceil(num_img/batch_size)
        epoch_loss = []
        for i in tqdm(range(n)):
            '''
            s_images  = [img for img in short_ex_img if img.startswith(name) and img.endswith('.ARW')]
            image = s_images[np.random.randint(0, len(s_images))]
            s_path = short_ex_img_path + image
            l_path = long_ex_img_path + [img for img in long_ex_img if img.startswith(name) and img.endswith('.ARW')][0]

            amp_ratio = get_amplification_ratio(s_path, l_path)
            s_img, l_img = load_image(s_path, l_path, name, amp_ratio)
            s_img, l_img = preprocess(s_img, l_img, amp_ratio)
            '''
            bs = min((i+1)*batch_size, num_img)-i*batch_size
            s_img, l_img = loader.make_batch(bs, order[i*batch_size:(i+1)*batch_size])

            optimizer.zero_grad()
            with autocast():
                s_img = model(s_img)
                loss_img = loss(s_img, l_img)
            
            scaler.scale(loss_img).backward()
            
            epoch_loss.append(loss_img.item())
            #print(loss_img)
            #loss_img.backward()
            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            
        e_loss = sum(epoch_loss)/n
        f = open(saved_model_param+'/train_loss.txt', 'a')
        f.write("\n{:4d} \t{:.5f}".format(epoch, e_loss))
        f.close()
        if epoch%save_every==0:
            torch.save(model.state_dict(), os.path.join(saved_model_param, 'epoch-{}.pth'.format(epoch)))
        #print('\n')
# WAŻNE: przy ładowaniu modelu musiałam dodać pole: MAP_LOCATION, trzeba zmienić przy używaniu gpu
def test_sony(model_name=U_net, loss=nn.MSELoss(reduction='mean'), locator='cpu'):
    model_loss = []
    order = []
    ratio_order = []
    result_dir = './result_Sony/'
    test_names = get_test_names()
    short_ex_img_path = './dataset/Sony/short/'
    long_ex_img_path = './dataset/Sony/long/'
    saved_prev, last_epoch, path_last_epoch = get_last_epoch()
    if saved_prev:
        model = model_name()
        model.load_state_dict(torch.load(saved_model_param + path_last_epoch, map_location=locator))
        model = model.to(device)
        model.eval()
    for name, param in model.named_parameters():
        if param.requires_grad:
            pass
    short_ex_img = os.listdir(short_ex_img_path)
    long_ex_img = os.listdir(long_ex_img_path)
    with torch.no_grad():
        for name in test_names:
            s_images  = [img for img in short_ex_img if img.startswith(name) and img.endswith('.ARW')]
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
                  result_dir + 'final/%s_%d_out.png' % (name_num, ratio))
                Image.fromarray(np.floor(np.clip(scale_full*255, 0, 256)).astype(np.uint8), mode='RGB').save(
                  result_dir + 'final/%s_%d_scale.png' % (name_num, ratio))
                Image.fromarray(np.floor(np.clip(long_img * 255, 0, 256)).astype(np.uint8), mode='RGB').save(
                  result_dir + 'final/%s_%d_gt.png' % (name_num, ratio))
    return model_loss, order, ratio_order


def plot_min_max_loss(model, metric, min_loss, min_, min_ratio,max_loss, max_, max_ratio, fig_x=20, fig_y=10):
    model_path = "result_Sony/final/"
    short_img = 'dataset/Sony/short/'
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(fig_x, fig_y))
    [axi.set_axis_off() for axi in axes.ravel()]
    axes[0][0].title.set_text('Short exposure')
    axes[0][1].title.set_text('Long exposure')
    axes[0][2].title.set_text('Model output')
    axes[0][3].title.set_text('Scaled output')
    name = min_[:9]
    gt_raw = rawpy.imread(short_img+min_)
    im = gt_raw.postprocess()
    axes[0][0].imshow(im)
    img = plt.imread(model_path+name+min_ratio+'_gt.png')
    axes[0][1].imshow(img)
    img = plt.imread(model_path+name+min_ratio+'_out.png')
    axes[0][2].imshow(img)
    img = plt.imread(model_path+name+min_ratio+'_scale.png')
    axes[0][3].imshow(img)
    
    name = max_[:9]
    gt_raw = rawpy.imread(short_img+max_)
    im = gt_raw.postprocess()
    axes[1][0].imshow(im)
    img = plt.imread(model_path+name+max_ratio+'_gt.png')
    axes[1][1].imshow(img)
    img = plt.imread(model_path+name+max_ratio+'_out.png')
    axes[1][2].imshow(img)
    img = plt.imread(model_path+name+max_ratio+'_scale.png')
    axes[1][3].imshow(img)
    plt.figtext(0.5,0.95, 'Minimal loss, model: %s, metric: %s, loss value: %.6f., ratio: %s' %(model, metric, min_loss, min_ratio), ha="center", va="top", fontsize=14, color="r")
    plt.figtext(0.5,0.5, 'Maximal loss, model: %s, metric: %s, loss value: %.6f, ratio: %s'%(model, metric, max_loss, max_ratio), ha="center", va="top", fontsize=14, color="r")


def plot_test_info(model, metric, model_loss, order, ratio_order):
    x = list(map(float, model_loss))
    ratio_order = list(map(str, map(int, ratio_order)))
    plt.hist(x, facecolor='g', alpha=0.75)
    plt.xlabel(metric)
    plt.ylabel('number of samples in bin')
    plt.title('Histogram of loss between short and long exposure image, model: %s' %model)
    plt.grid(True)
    plt.show()

    min_loss, min_, min_ratio = min(x), order[x.index(min(x))], str(int(ratio_order[x.index(min(x))]))
    max_loss, max_, max_ratio = max(x), order[x.index(max(x))], str(int(ratio_order[x.index(max(x))]))
    plot_min_max_loss(model, metric, min_loss, min_, min_ratio,max_loss, max_, max_ratio)

def plot_train_loss(model_name, path):
    f = open(path, 'r')
    epoch = []
    train_loss = []
    for line in f:
        try:
            epoch.append(int(line.split()[0]))
            train_loss.append(float(line.split()[1]))
        except:
            continue
    f.close()
    plt.plot(epoch, train_loss, label='train loss')
    plt.title(model_name)
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

#train_sony(model, get_train_names(), 201, 5)
test_sony(ConvDeconv)

'''
plot_train_loss('U-net', r'C:\Users\kapolak\Documents\GitHub\Hello-Darkness\models/train_loss.txt')
model_loss, order, ratio_order = test_sony()
plot_test_info('U-net', 'MSELoss', model_loss, order, ratio_order)
'''
