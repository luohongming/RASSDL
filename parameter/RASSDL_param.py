import argparse


parser = argparse.ArgumentParser(description='TM_RDN_model_batch parameters')

parser.add_argument('--gpu-ids', type=list, default=[0], help='for example, [0, 1] [1, 3, 4] None')
parser.add_argument('--bins-num', type=int, default=256, help='number of bins')
parser.add_argument('--seed', type=int, default=35, help='random seed to use')

parser.add_argument('--n-dense', type=int, default=6, help='length of dense block')
parser.add_argument('--nf', type=int, default=64, help='n features')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--eta-min', type=float, default=1e-6, help='min learning rate')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')

parser.add_argument('--save-model-dir', type=str, default='./checkpoints', help='dir to save model')
parser.add_argument('--pretrain-model-path', type=str, default='', help='path of pretrained model')
parser.add_argument('--network-label', type=str, default='TM_RDN_batch', help='network label')
parser.add_argument('--dataset-path', type=str, default='./data/HDRPS_log_64.h5', help='path to hdr image')

parser.add_argument('--dataset-name', type=str, default='test')
parser.add_argument('--size', type=int, default=64)
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--mat-path', type=str, default='')
parser.add_argument('--mat-name', type=str, default='')
parser.add_argument('--save-path', type=str, default='')