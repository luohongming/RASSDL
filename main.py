#  Ubuntu 16.04
#                 python = 3.7
#                  numpy = 1.16.4
#  opencv-contrib-python = 4.2.0.34
#                  scipy = 1.3.0
#                  torch = 1.0.1.post2
#            torchvision = 0.2.2.post3


import os
from parameter.RASSDL_param import *


def add_backslash(str_name):
    l = len(str_name)
    new_name = ''
    for i in range(l):
        if str_name[i] == '(':
            new_name = new_name + '\('
        elif str_name[i] == ')':
            new_name = new_name + '\)'
        else:
            new_name = new_name + str_name[i]
    return new_name


if __name__ == '__main__':
    opt = parser.parse_args()

    epoch_label = opt.nEpochs
    size = opt.size
    scale = opt.scale
    save_model_dir = opt.save_model_dir
    dataset_name = opt.dataset_name
    data_dir = './{}/'.format(dataset_name)
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            mat_name = file[0:-4]
            mat_name = add_backslash(mat_name)
            mat_file = mat_name + '.mat'
            mat_path = os.path.join(root, mat_file)
            print(mat_path)

            commend = 'python main_train.py --mat-path={} --mat-name={}'.format(mat_path, mat_name)
            os.system(commend)

            save_path = '{}/{}/{}_{}_{}.pth'.format(save_model_dir, mat_name, epoch_label, size, scale)
            commend = 'python main_test.py --mat-path={} --mat-name={} --save-path={}'.format(mat_path, mat_name, save_path)
            os.system(commend)