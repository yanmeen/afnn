# -*- coding: utf-8 -*-

# =============================================================================
# by Dr. Ming Yan (10/2018)
# yan.meen@gmail.com
#
# modified on the code from  https://github.com/cszn
# =============================================================================

# no need to run this code separately


import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from keras.utils import np_utils
aug_Times = 8


def show(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation="nearest", cmap="gray")
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):

    if mode == 0:
        img_a = img
    elif mode == 1:
        img_a = np.flipud(img)
    elif mode == 2:
        img_a = np.rot90(img)
    elif mode == 3:
        img_a = np.flipud(np.rot90(img))
    elif mode == 4:
        img_a = np.rot90(img, k=2)
    elif mode == 5:
        img_a = np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        img_a = np.rot90(img, k=3)
    elif mode == 7:
        img_a = np.flipud(np.rot90(img, k=3))

    return img_a


def gen_data(file_name):

    # read image
    imgs = skio.imread(file_name)  # gray scale
    f, h, w = imgs.shape
    patches = []
    labels = []
    y = []

    for i in range(0, 50):
        y.extend([i, i, i, i])

    y = np.array(y, dtype="float32")
    #   y = y / 100

    for k in range(0, aug_Times):
        for s in range(0, f, 1):
            # extract patches
            x = imgs[s]
            # filter pixels that value < 8 as background noise
            x[x < 8] = 0
            # data augmentation
            x_aug = data_aug(x, mode=np.random.randint(0, 8))
            x_aug = x_aug.astype("float32")
            x_aug = x_aug / 255
            patches.append(x_aug)
            labels.append(y[s])

    return patches, labels


def data_generator(data_dir="data/SMRes128", verbose=False):

    # get name list of all .png files
    file_list = glob.glob(data_dir + "/*.tif")
    # initrialize
    data = []
    data_label = []
    # generate patches
    for i in range(len(file_list)):
        patch, label = gen_data(file_list[i])
        data.extend(patch)
        data_label.extend(label)
        if verbose:
            print(str(i + 1) + "/" + str(len(file_list)) + " is done ^_^")
    data = np.array(data, dtype="uint8")
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
    data_label = np.array(data_label, dtype="float32")
    data_label = np_utils.to_categorical(data_label)
    print("^_^-training data finished-^_^")
    return data, data_label


if __name__ == "__main__":

    data, label = data_generator(data_dir="data/Res128")


#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')
