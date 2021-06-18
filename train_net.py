import torch
from torch import nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from Loader import Loader
from tqdm import tqdm
import os
import math
import numpy as np

def train_sony(model, train_names, saved_model_param, num_epoch, batch_size=1, save_every=10, loss = nn.L1Loss(reduction='mean', loader=Loader())):
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    model = model.train()
    scaler = GradScaler()
    
    for epoch in range(0, num_epoch):
        if epoch > 2000:
            for param in optimizer.param_groups:
                param['lr'] = 1e-5
        order = np.random.permutation(train_names)
        num_img = len(order)
        n = math.ceil(num_img/batch_size)
        epoch_loss = []
        for i in tqdm(range(n)):
            bs = min((i+1)*batch_size, num_img)-i*batch_size
            s_img, l_img = loader.make_batch(bs, order[i*batch_size:(i+1)*batch_size])

            optimizer.zero_grad()
            with autocast():
                s_img = model(s_img)
                loss_img = loss(s_img, l_img)
            
            scaler.scale(loss_img).backward()
            
            epoch_loss.append(loss_img.item())
            scaler.step(optimizer)
            scaler.update()
            
        e_loss = sum(epoch_loss)/n
        f = open(saved_model_param+'/train_loss.txt', 'a')
        f.write("\n{:4d} \t{:.5f}".format(epoch, e_loss))
        f.close()
        if epoch%save_every==0:
            torch.save(model.state_dict(), os.path.join(saved_model_param, 'epoch-{}.pth'.format(epoch)))