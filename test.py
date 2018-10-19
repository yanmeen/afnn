# -*- coding: utf-8 -*-

# =============================================================================
# by Dr. Ming Yan (10/2018)
# yan.meen@gmail.com
# https://github.com/yanmeen/afnn
# modified on the code from https://github.com/cszn
# =============================================================================

# run this to test the model

import argparse
import os
import time
import datetime
# import PIL.Image as Image
import numpy as np
from keras.models import load_model, model_from_json
# from skimage.measure import compare_psnr, compare_ssim
from skimage import io as skio
from keras.utils import np_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/TestRes128',
                        type=str, help='directory of test dataset')
    parser.add_argument(
        '--set_names', default=['SM', 'MM'], type=list, help='name of test dataset')
    parser.add_argument('--model_dir', default=os.path.join('trained',
                                                            'AFNN_128'), type=str, help='directory of the model')
    parser.add_argument('--model_name', default='model.hdf5',
                        type=str, help='the model name')
    parser.add_argument('--result_dir', default='results',
                        type=str, help='directory of results')
    parser.add_argument('--save_result', default=0, type=int,
                        help='save the results, 1 or 0')
    return parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


if __name__ == '__main__':

    args = parse_args()

    # =============================================================================
    #     # serialize model to JSON
    #     model_json = model.to_json()
    #     with open("model.json", "w") as json_file:
    #         json_file.write(model_json)
    #     # serialize weights to HDF5
    #     model.save_weights("model.h5")
    #     print("Saved model")
    # =============================================================================

    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):
        # load json and create model
        json_file = open(os.path.join(args.model_dir, 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(os.path.join(args.model_dir, 'model.h5'))
        log('load trained model')
    else:
        model = load_model(os.path.join(
            args.model_dir, args.model_name), compile=False)
        log('load trained model')

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:

        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))

        error_list = []
        pred_list = []
        for im in os.listdir(os.path.join(args.set_dir, set_cur)):
            if im.endswith(".tif"):
                # x = np.array(Image.open(os.path.join(args.set_dir,set_cur,im)), dtype='float32') / 255.0
                # read image
                imgs = skio.imread(os.path.join(
                    args.set_dir, set_cur, im))  # gray scale
                focal_number = imgs.shape[0]
                data = []
                #grd_truth = []

                for id in range(0, focal_number):
                    x = np.array(imgs[id], dtype=np.float32) / 255.0
                    x = x.reshape(x.shape[0], x.shape[1], 1)
                    data.append(x)
                    # grd_truth.append(id//4)

                start_time = time.time()
                data = np.array(data, dtype="float32")
                data = data.reshape(
                    data.shape[0], data.shape[1], data.shape[2], 1)
                #grd_truth = np_utils.to_categorical(grd_truth)

                pred = model.predict(
                    data, batch_size=200, verbose=0)  # inference
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' %
                      (set_cur, im, elapsed_time))
                #test_error = grd_truth - pred
                # print('The test error is : ', test_error)
                # error_list.append(test_error)
                pred_list.append(pred)
                # acc_list.append(acc)
                # np.savetxt(os.path.join(args.result_dir, set_cur,
                #                        'test_error.csv'), test_error, delimiter=",")
                np.savetxt(os.path.join(args.result_dir, set_cur,
                                        'prediction.csv'), pred, delimiter=",")

        #error_avg = np.mean(error_list)
        #print('The verage error is : ', error_avg)

        if args.save_result:
            # np.savetxt(os.path.join(
            #    args.result_dir, set_cur, 'results_errorlist.txt'), error_list, delimiter=",")
            np.savetxt(os.path.join(
                args.result_dir, set_cur, 'results_predlist.txt'), pred_list, delimiter=",")
