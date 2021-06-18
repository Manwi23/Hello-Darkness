import matplotlib.pyplot as plt
import rawpy
import os
import torch
from torch import nn

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


#czemu nie widać podpisów???
def plot_k_model_outputs_with_loss(k=3, loss=nn.MSELoss(reduction='mean'), fig_x=20, fig_y=10):
    img_list = os.listdir("result_Sony/final")
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
        name = img_list[i*3][:-10]
        gt_raw = rawpy.imread(short_img+name+'0.04s.ARW')
        im = gt_raw.postprocess()
        axes[i][0].imshow(im)
        img = plt.imread(model_path+img_list[i*3])
        loss_values.append(float(loss(torch.tensor(im), torch.tensor(img))))
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