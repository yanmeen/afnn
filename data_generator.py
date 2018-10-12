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
from skimage import io as skio

aug_times = 4

def show(x,title=None,cbar=False,figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_name):

    # read image
    imgs = skio.imread(file_name)  # gray scale
    f, h, w = imgs.shape
    patches = []
	patches_labels = []
	y = list(range(-100, 100, 1))
	y = np.array(y, dtype='float32')
	y = y/100
    for s in range(0, f, 1):
        # extract patches
		x = imgs[s]
        # data aug
        for k in range(0, aug_times):
            x_aug = data_aug(x, mode=np.random.randint(0,8))
            patches.append(x_aug)
			patches_labels.append(y)
                
    return patches, patches_labels

def datagenerator(data_dir='data/Res128',verbose=False):
    
    file_list = glob.glob(data_dir+'/*.tif')  # get name list of all .png files
    # initrialize
    data = []
	data_label = []
    # generate patches
    for i in range(len(file_list)):
        patch, label = gen_patches(file_list[i])
        data.append(patch)
		data_label.append(label)
        if verbose:
            print(str(i+1)+'/'+ str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    #data = data.reshape((data.shape[0]*data.shape[1],data.shape[2],data.shape[3],1))
    #discard_n = len(data)-len(data)//batch_size*batch_size;
    #data = np.delete(data,range(discard_n),axis = 0)
    print('^_^-training data finished-^_^')
    return data, data_label

if __name__ == '__main__':   

    data, label = datagenerator(data_dir='data/Res128')
    

#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')       