import matplotlib.pyplot as plt
import rawpy
import os
import torch
from torch import nn
from statistics import mean

def plot_k_model_outputs(model, metric, k=3, fig_x=20, fig_y=10):
    img_list = os.listdir("result_Sony/final")
    img_list = [x for x in img_list if model in x and metric in x]
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
        name = img_list[i*3][:9]
        try:
            gt_raw = rawpy.imread(short_img+name+'0.04s.ARW')
        except:
            gt_raw = rawpy.imread(short_img+name+'0.1s.ARW')
        im = gt_raw.postprocess()
        axes[i][0].imshow(im)
        img = plt.imread(model_path+img_list[i*3])
        axes[i][1].imshow(img)
        img = plt.imread(model_path+img_list[i*3+1])
        axes[i][2].imshow(img)
        img = plt.imread(model_path+img_list[i*3+2])
        axes[i][3].imshow(img)


#czemu nie widać podpisów???
def plot_k_model_outputs_with_loss(model, metric, k=3, loss=nn.MSELoss(reduction='mean'), fig_x=20, fig_y=10):
    img_list = os.listdir("result_Sony/final")
    img_list = [x for x in img_list if model in x and metric in x]
    img_list.sort()
    model_path = "result_Sony/final/"
    short_img = 'dataset/Sony/short/'
    fig, axes = plt.subplots(nrows=k, ncols=5, figsize=(fig_x, fig_y))
    [axi.set_axis_off() for axi in axes.ravel()]
    axes[0][0].title.set_text('Short exposure')
    axes[0][1].title.set_text('Long exposure')
    axes[0][2].title.set_text('Model output')
    axes[0][3].title.set_text('Scaled output')
    axes[0][4].title.set_text('Loss between')
    between_names = ['short vs long', 'model vs long', 'scaled vs long']
    for i in range(k):
        loss_values = []
        name = img_list[i*3][:9]
        try:
            gt_raw = rawpy.imread(short_img+name+'0.04s.ARW')
        except:
            gt_raw = rawpy.imread(short_img+name+'0.1s.ARW')
        im = gt_raw.postprocess()
        axes[i][0].imshow(im)
        img = plt.imread(model_path+img_list[i*3])
        loss_values.append(float(loss(torch.tensor(im.astype(float)), torch.tensor(img.astype(float)))))
        axes[i][1].imshow(img)
        img2 = plt.imread(model_path+img_list[i*3+1])
        loss_values.append(float(loss(torch.tensor(img), torch.tensor(img2))))
        axes[i][2].imshow(img2)
        img3 = plt.imread(model_path+img_list[i*3+2])
        axes[i][3].imshow(img3)
        loss_values.append(float(loss(torch.tensor(img), torch.tensor(img3))))
        axes[i][4].bar(between_names, loss_values, log=True)


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


def plot_train_loss_for_metric(metric, path, log_scale=True):
    logs = [x for x in os.listdir(path) if metric in x]
    for log in logs:
        f = open(path+'\\' +log , 'r')
        epoch = []
        train_loss = []
        for line in f:
            try:
                epoch.append(int(line.split()[0]))
                train_loss.append(float(line.split()[1]))
            except:
                continue
        f.close()
        plt.plot(epoch, train_loss, label=log.split('.')[0][:-len(metric)])
    plt.title('Train loss for metric: %s, scale_log: %r' %(metric, log_scale))
    plt.xlabel('epoch')
    if log_scale:
        plt.yscale("log")
    plt.legend()
    plt.show()


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
    img = plt.imread(model_path+name+min_ratio+'_%s_' % model+metric+'_gt.png')
    axes[0][1].imshow(img)
    img = plt.imread(model_path+name+min_ratio+'_%s_' % model+metric+'_out.png')
    axes[0][2].imshow(img)
    img = plt.imread(model_path+name+min_ratio+'_%s_' % model+metric+'_scale.png')
    axes[0][3].imshow(img)
    
    name = max_[:9]
    gt_raw = rawpy.imread(short_img+max_)
    im = gt_raw.postprocess()
    axes[1][0].imshow(im)
    img = plt.imread(model_path+name+max_ratio+'_%s_' % model+metric+'_gt.png')
    axes[1][1].imshow(img)
    img = plt.imread(model_path+name+max_ratio+'_%s_' % model+metric+'_out.png')
    axes[1][2].imshow(img)
    img = plt.imread(model_path+name+max_ratio+'_%s_' % model+metric+'_scale.png')
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


def compare_models_outputs(image_name, fig_x=20, fig_y=10):
    loss1=nn.L1Loss(reduction='mean')
    loss2=nn.MSELoss(reduction='mean')
    img_list = os.listdir("result_Sony/final")
    img_list = [x for x in img_list if image_name in x and 'out' in x]
    img_list.sort()
    l_img = img_list[0][:-7]+'gt.png'
    model_path = "result_Sony/final/"
    short_img = 'dataset/Sony/short/'
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(fig_x, fig_y))
    [axi.set_axis_off() for axi in axes.ravel()]
    axes[0][0].title.set_text('Long exposure')
    img = plt.imread(model_path+l_img)
    axes[0][0].imshow(img)
    s_img = img_list[0][:9]
    try:
        gt_raw = rawpy.imread(short_img+s_img+'0.04s.ARW')
    except:
        gt_raw = rawpy.imread(short_img+s_img+'0.1s.ARW')
    img1 = gt_raw.postprocess()
    l1 = float(loss1(torch.tensor(img.astype(float)), torch.tensor(img1.astype(float))))
    l2 = float(loss2(torch.tensor(img), torch.tensor(img1)))
    axes[0][1].imshow(img1)
    axes[0][1].title.set_text('Short exposure, L1:  %.6f, MSE:  %.6f' %(l1, l2))
    img2 = plt.imread(model_path+img_list[0])
    axes[0][2].imshow(img2)
    l = float(loss1(torch.tensor(img), torch.tensor(img2)))
    axes[0][2].title.set_text('ConvDeconv_L1, L1:  %.6f' %l)
    img3 = plt.imread(model_path+img_list[1])
    l = float(loss2(torch.tensor(img), torch.tensor(img3)))
    axes[0][3].imshow(img3)
    axes[0][3].title.set_text('ConvDeconv_MSE, MSE:  %.6f' %l)
    
    img4 = plt.imread(model_path+img_list[2])
    axes[1][0].imshow(img4)
    l = float(loss1(torch.tensor(img), torch.tensor(img4)))
    axes[1][0].title.set_text('ResNet_L1, L1:  %.6f' %l)
    img5 = plt.imread(model_path+img_list[3])
    l = float(loss2(torch.tensor(img), torch.tensor(img5)))
    axes[1][1].imshow(img5)
    axes[1][1].title.set_text('ResNet_MSE, MSE:  %.6f' %l)
    img6 = plt.imread(model_path+img_list[4])
    l = float(loss1(torch.tensor(img), torch.tensor(img6)))
    axes[1][2].imshow(img6)
    axes[1][2].title.set_text('U_net_L1, L1:  %.6f' %l)
    img7 = plt.imread(model_path+img_list[5])
    l = float(loss2(torch.tensor(img), torch.tensor(img7)))
    axes[1][3].imshow(img7)
    axes[1][3].title.set_text('U_net_MSE, MSE:  %.6f' %l)


def plot_mean_test_loss(metric_name, loss):
    model_path = "result_Sony/final/"
    convdeconv, resnet, unet = [], [], []
    img_list = os.listdir("result_Sony/final")
    out_list = [x for x in img_list if metric_name in x and 'out' in x]
    gt_list = [x for x in img_list if metric_name in x and 'gt' in x]
    out_list.sort()
    gt_list.sort()
    for i in range(0, len(out_list), 3):
        gt1 = plt.imread(model_path+gt_list[i])
        out1 = plt.imread(model_path+out_list[i])
        convdeconv.append(float(loss(torch.tensor(gt1), torch.tensor(out1))))
        gt2 = plt.imread(model_path+gt_list[i+1])
        out2 = plt.imread(model_path+out_list[i+1])
        resnet.append(float(loss(torch.tensor(gt2), torch.tensor(out2))))
        gt3 = plt.imread(model_path+gt_list[i+2])
        out3 = plt.imread(model_path+out_list[i+2])
        unet.append(float(loss(torch.tensor(gt3), torch.tensor(out3))))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    nets = ['ConvDeconv', 'ResNet', 'Unet']
    means = [mean(convdeconv), mean(resnet), mean(unet)]
    ax.bar(nets,means)
    plt.title('Mean test loss for %s' %metric_name)
    plt.show()