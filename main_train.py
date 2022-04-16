import numpy as np
from queue import Queue
import cv2
from threading import Thread
import h5py
from dataset import DatasetFromHdf5
from torch.utils.data import DataLoader
import scipy.io as sio
from model.RASSDL_model import RASSDL_model
from parameter.RASSDL_param import *
import torch
import torch.backends.cudnn as cudnn
import os
import time


def cal_k(hdr_y):
    eps = 1e-8
    I = hdr_y
    Imax = I.max()
    Imin = I.min()
    Iave = np.exp(np.log(I + eps).mean())

    A = 0.4
    B = 1.2
    k = A * np.power(B, (np.log(Iave + eps) * 2 - np.log(Imin + eps) - np.log(Imax + eps)) / (
                np.log(Imax + eps) - np.log(Imin + eps)))
    return k


def cal_tao(hdr_y, k):
    I = hdr_y
    Imax = I.max()
    Imin = I.min()

    tao_0 = 1e-8
    num = 0
    while(True):
        num += 1
        tmp1 = (1 / (I + tao_0) - 1 / (Imin + tao_0)) * (np.log(Imax + tao_0) - np.log(Imin + tao_0))
        tmp2 = (np.log(I + tao_0) - np.log(Imin + tao_0)) * (1 / (Imax + tao_0) - 1 / (Imin + tao_0))
        tmp3 = np.power((np.log(Imax + tao_0) - np.log(Imin + tao_0)), 2)
        f_dao = ((tmp1 - tmp2) / tmp3).mean()
        f = ((np.log(I + tao_0) - np.log(Imin + tao_0)) / (np.log(Imax + tao_0) - np.log(Imin + tao_0))).mean() - k
        tao_1 = tao_0 - (f / f_dao)
        if np.abs(tao_0 - tao_1) < 1e-8 or num >= 20:
            if np.isnan(tao_1):
                exit('tao error!')
            else:
                break
        tao_0 = tao_1
    tao = tao_1
    return tao


def add_data(h5_file, mat_path, height, width, max_images, nums, scale, stride):
    mat_list = list([mat_path])
    num_mats = len(mat_list)
    scale = int(scale)
    height = int(height)
    width = int(width)

    dset_name1 = 'original'
    dset_name2 = 'equlized'
    dset_size = (num_mats * nums, 1, height, width)
    imgs_dset1 = h5_file.create_dataset(dset_name1, dset_size, np.float32)
    imgs_dset2 = h5_file.create_dataset(dset_name2, dset_size, np.float32)

    input_queue = Queue()
    output_queue = Queue()

    def read_worker():
        while True:
            idx, filename = input_queue.get()
            hdr = sio.loadmat(filename)['hdr']
            hdr = np.ascontiguousarray(hdr[:, :, ::-1], dtype=np.float32)
            if hdr.min() < 0:
                hdr = hdr - hdr.min()
            hdr_y = 0.299 * hdr[:, :, 0] + 0.587 * hdr[:, :, 1] + 0.114 * hdr[:, :, 2]
            hdr_y = hdr_y - hdr_y.min() + 1e-8
            k = cal_k(hdr_y)
            tao = cal_tao(hdr_y, k)
            hdr_y_tao = hdr_y + tao + 1e-8
            log_y = (np.log(hdr_y_tao) - np.log(hdr_y_tao.min())) / (np.log(hdr_y_tao.max()) - (np.log(hdr_y_tao.min())))
            H, W = log_y.shape[0], log_y.shape[1]

            h = 0
            w = 0
            idex = 0
            while h <= (H - (height * scale)):
                while w <= (W - (width * scale)):
                    try:
                        x1 = h
                        y1 = w
                        img_tmp = log_y[x1:(x1 + (height * scale)), y1:(y1 + (width * scale))]
                        img = img_tmp[::scale, ::scale]
                    except (ValueError, ImportError) as e:
                        print(hdr)
                        print(hdr.shape, hdr.dtype)
                        print(e)
                    output_queue.put((idex, img))
                    idex = idex + 1
                    w = w + stride
                w = 0
                h = h + stride
            input_queue.task_done()

    def write_worker():
        num_written = 0
        while True:
            idx, img = output_queue.get()
            if img.ndim == 3:
                imgs_dset1[idx] = img.transpose(2, 0, 1)

                img_equlized = np.array(img * 255, dtype=np.uint8)
                img_equlized = cv2.equalizeHist(img_equlized)
                img_equlized = np.array(img_equlized / 255, dtype=np.float32)
                imgs_dset2[idx] = img_equlized
            elif img.ndim == 2:
                imgs_dset1[idx] = img

                img_equlized = img[..., np.newaxis]
                img_equlized = np.array(img_equlized * 255, dtype=np.uint8)
                img_equlized = cv2.equalizeHist(img_equlized)
                img_equlized = np.array(img_equlized / 255, dtype=np.float32)
                imgs_dset2[idx] = img_equlized
            output_queue.task_done()
            num_written = num_written + 1
            # if num_written % 10 == 0:
            #     print('Copied {} / {}'.format(num_written, num_mats * nums))

    t = Thread(target=read_worker)
    t.daemon = True
    t.start()

    t = Thread(target=write_worker)
    t.daemon = True
    t.start()

    for idx, filename in enumerate(mat_list):
        if max_images > 0 and idx >= max_images: break
        input_queue.put((idx, filename))

    input_queue.join()
    output_queue.join()


def create_h5(mat_path, mat_name, dataset, size, scale):
    if not os.path.exists('./data/'):
        os.mkdir('./data/')
    if not os.path.exists('./data/{}/'.format(dataset)):
        os.mkdir('./data/{}/'.format(dataset))

    output_file = './data/{}/{}_{}_{}.h5'.format(dataset, mat_name, size, scale)
    height = size
    width = size
    max_images = -1
    stride = int(size * scale / 2)

    hdr = sio.loadmat(mat_path)['hdr']
    hdr = np.ascontiguousarray(hdr[:, :, ::-1], dtype=np.float32)
    H, W = hdr.shape[0], hdr.shape[1]
    nums = int((H - (height * scale - 1) + stride - 1) / stride) * int((W - (width * scale - 1) + stride - 1) / stride)

    with h5py.File(output_file, 'w') as f:
        add_data(f, mat_path, height, width, max_images, nums, scale, stride)
    return output_file


if __name__ == '__main__':
    opt = parser.parse_args()
    cudnn.benchmark = True

    mat_path = opt.mat_path
    mat_name = opt.mat_name
    dataset_name = opt.dataset_name
    size = opt.size

    scale = 1
    h5_path = create_h5(mat_path, mat_name, dataset_name, size, scale)
    DataSet = DatasetFromHdf5(h5_path)

    model = RASSDL_model(opt)
    model.set_mode(train=True, save_path=opt.pretrain_model_path)

    loader = DataLoader(DataSet, batch_size=opt.batch_size, shuffle=True)
    current_step = 0
    epoch = 0
    time_start = time.time()
    for epoch in range(1, opt.nEpochs + 1):
        for i, data in enumerate(loader, 1):
            model.set_input(data)
            model.set_target()
            current_step += 1
            model.train()
            lr = model.get_current_learning_rate()
            if i % 4 == 0:
                print('epoch: {}, current_step: {}, loss: {}, lr: {}'.format(epoch, current_step, model.loss.item(), lr))
        model.update_learning_rate(current_step)
    time_end = time.time()
    print('running time: {}'.format(time_end-time_start))
    save_path = model.save(mat_name, epoch, size, scale)

