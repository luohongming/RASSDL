import scipy.io as sio
from model.RASSDL_model import RASSDL_model
from torchvision.utils import save_image
from utils import set_color_map
import torch
import torch.backends.cudnn as cudnn
from parameter.RASSDL_param import *
import os
import time


if __name__ == '__main__':
    opt = parser.parse_args()
    torch.manual_seed(opt.seed)
    cudnn.benchmark = True

    size = opt.size
    hdr_path = opt.mat_path
    save_path = opt.save_path
    dataset_name = opt.dataset_name
    mat_name = opt.mat_name

    hdr = sio.loadmat(hdr_path)['hdr']
    hdr = torch.from_numpy(hdr).float()
    if hdr.min() < 0:
        hdr = hdr - hdr.min()
    [hdr_r, hdr_g, hdr_b] = hdr.chunk(3, 2)
    hdr_y = 0.299 * hdr_r + 0.587 * hdr_g + 0.114 * hdr_b
    hdr_y = hdr_y - hdr_y.min() + 1e-8

    hdr_y = hdr_y.permute(2, 0, 1).unsqueeze(0)
    hdr_r = hdr_r.permute(2, 0, 1).unsqueeze(0)
    hdr_g = hdr_g.permute(2, 0, 1).unsqueeze(0)
    hdr_b = hdr_b.permute(2, 0, 1).unsqueeze(0)

    # model
    model = RASSDL_model(opt)
    model.set_mode(train=False, save_path=save_path)
    hdr_y = hdr_y.to(model.device)
    hdr_r = hdr_r.to(model.device)
    hdr_g = hdr_g.to(model.device)
    hdr_b = hdr_b.to(model.device)
    model.set_eval_input(hdr_y)
    time_start = time.time()
    with torch.no_grad():
        output = model.model(model.input)
    time_end = time.time()
    print('running time: {}'.format(time_end-time_start))
    ldr_output = set_color_map(hdr_y, hdr_r, hdr_g, hdr_b, output)
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    if not os.path.exists('./results/{}/'.format(dataset_name)):
        os.mkdir('./results/{}/'.format(dataset_name))
    img_save_path = './results/test/{}.png'.format(mat_name)
    save_image(ldr_output, img_save_path)
