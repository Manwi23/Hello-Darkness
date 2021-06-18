import os


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


def get_last_epoch():
    saved_model_param = 'models/Sony5'

    def get_epoch(path):
        if "epoch" in path:
            return int(path.split('.')[0].split('-')[-1])
        else:
            return -1

    try:
        last_epoch = max(map(get_epoch, os.listdir(saved_model_param)))
    except:
        return False, 0, None

    return True, last_epoch, '/epoch-%d.pth' % last_epoch