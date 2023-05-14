#This script is to test the whole data flow
import argparse
import contextlib
import os
import random
import torch.nn as nn
import torch
import torch.optim as optim
from datasets.dataset_h5 import all_h5
from models.merger_net import Net
from models.composed_chamfer import composed_sqrt_chamfer
arg_parser = argparse.ArgumentParser(description="Training keypoint reconstruction",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-t', '--train-data-dir', type=str, default='./data/point_cloud/train',
                        help='Directory that contains training .h5 files.')
arg_parser.add_argument('-v', '--val-data-dir', type=str, default='./data/point_cloud/val',
                        help='Directory that contains validation .h5 files.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size.') #8
arg_parser.add_argument('-c', '--subclass', type=int, default=14,
                        help='Subclass label ID to train on.')  # 14 is `chair` class.
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='./models/check_points_h5/twosingle.pt',
                        help='Model checkpoint file path for saving.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=10,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda:0',
                        help='Pytorch device for training.')
arg_parser.add_argument('-e', '--epochs', type=int, default=80,
                        help='Number of epochs to train.')
arg_parser.add_argument('-n', '--name', type=str, default='self_reconstruction',
                        help='self/mutual_reconstruction')


def L2(embed):
    return 0.01 * (torch.sum(embed ** 2))



def train_merger_net(net, optimizer, x_set, shuffle, batch, epochs, checkpoint_path):
    for epoch in range(epochs):
        running_loss = 0.0
        running_lrc = 0.0
        running_ldiv = 0.0
        net.train()
        if shuffle:
            x_set = list(x_set)
            random.shuffle(x_set)
        # print("len of x_set:{}".format(len(x_set)))
        with contextlib.suppress():

            for i in range(len(x_set) // batch):
                idx = slice(i * batch, (i + 1) * batch)
                refp = next(net.parameters())
                batch_x = torch.tensor(x_set[idx], device=refp.device)
                # print("size of batch_x:{}".format(batch_x.shape))
                optimizer.zero_grad()
                RPCD, KPCD, KPA, LF, MA = net(batch_x)
                blrc = composed_sqrt_chamfer(batch_x, RPCD, MA)
                # print("len of RPCD:{}, MA:{}".format(len(RPCD),len(MA)))
                # print("shape of RPCD element:{} MA element1:{} MA element2: {}".format(RPCD[0].shape, MA[0].shape, MA[1].shape))
                # print("RPCD[0]:{}, MA[0]:{}, MA[1]:{}".format(RPCD[0], MA[0], MA[1]))
                bldiv = L2(LF)
                loss = blrc + bldiv
                loss.backward()
                optimizer.step()

                running_lrc += blrc.item()
                running_ldiv += bldiv.item()
                running_loss += loss.item()
                print('[%s%d, %4d] loss: %.4f Lrc: %.4f Ldiv: %.4f' %
                ('VT'[True], epoch, i, running_loss / (i + 1), running_lrc / (i + 1), running_ldiv / (i + 1)))

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
        }, checkpoint_path)


def train_mutual_net(net, optimizer, x_set, shuffle, batch, epochs, checkpoint_path):
    print("In mutual mode")
    for epoch in range(epochs):
        running_loss = 0.0
        running_lrc_self = 0.0
        running_lrc_mutual = 0.0
        running_ldiv = 0.0
        net.train()
        if shuffle:
            x_set = list(x_set)
            random.shuffle(x_set)
        # print("len of x_set:{}".format(len(x_set)))

        #split training set
        split_num = len(x_set)//2
        x_set_a = x_set[0:split_num]
        x_set_b = x_set[split_num:2*split_num]

        with contextlib.suppress():
            for i in range(split_num // batch):
                idx = slice(i * batch, (i + 1) * batch)
                refp = next(net.parameters())
                batch_x_a = torch.tensor(x_set_a[idx], device=refp.device)
                batch_x_b = torch.tensor(x_set_b[idx], device=refp.device)
                # print("size of batch_x:{}".format(batch_x.shape))
                optimizer.zero_grad()
                RPCD_a, KPCD_a, KPA_a, LF_a, MA_a = net(batch_x_a)
                RPCD_b, KPCD_b, KPA_b, LF_b, MA_b = net(batch_x_b)
                blrc_a = composed_sqrt_chamfer(batch_x_a, RPCD_a, MA_a)
                blrc_b = composed_sqrt_chamfer(batch_x_b, RPCD_b, MA_b)

                blrc_ab = composed_sqrt_chamfer(batch_x_a, RPCD_b, MA_a)
                blrc_ba = composed_sqrt_chamfer(batch_x_b, RPCD_a, MA_b)

                blrc = blrc_ab + blrc_ba + blrc_a + blrc_b
                # print("len of RPCD:{}, MA:{}".format(len(RPCD),len(MA)))
                # print("shape of RPCD element:{} MA element1:{} MA element2: {}".format(RPCD[0].shape, MA[0].shape, MA[1].shape))
                # print("RPCD[0]:{}, MA[0]:{}, MA[1]:{}".format(RPCD[0], MA[0], MA[1]))
                bldiv_a = L2(LF_a)
                bldiv_b = L2(LF_b)

                loss = blrc + bldiv_a + bldiv_b
                loss.backward()
                optimizer.step()

                running_lrc_self += blrc_a.item() + blrc_b.item()
                running_lrc_mutual += blrc_ab.item() + blrc_ba.item()
                running_ldiv += bldiv_a.item() + blrc_b.item()
                running_loss += loss.item()
                print('[%s%d, %4d] loss: %.4f Lrc_s: %.4f Lrc_m: %.4f Ldiv: %.4f' %
                ('VT'[True], epoch, i, running_loss / (i + 1), running_lrc_self / (i + 1), running_lrc_mutual / (i + 1), running_ldiv / (i + 1)))

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
        }, checkpoint_path)


if __name__ == '__main__':
    args = arg_parser.parse_args()
    DATASET = args.train_data_dir
    VALSET = args.val_data_dir
    batch = args.batch
    x, xl = all_h5(DATASET, True, True, subclasses=(args.subclass,), sample=None)
    x_test, xl_test = all_h5(VALSET, True, True, subclasses=(args.subclass,), sample=None)
    print("x shape={}".format(x.shape))

    net = Net(args.max_points, args.n_keypoint).to(args.device)
    net = nn.DataParallel(net, device_ids=[0, 2]).module

    if args.name == 'self':

        optimizer = optim.Adadelta(net.parameters(), eps=1e-2)
        train_merger_net(net, optimizer, x, True, batch, args.epochs, args.checkpoint_path)

    elif args.name == 'mutual':
        print(os.path.exists(args.checkpoint_path))
        optimizer = optim.Adadelta(net.parameters(), eps=1e-2)
        train_mutual_net(net, optimizer, x, True, batch, args.epochs, args.checkpoint_path)

