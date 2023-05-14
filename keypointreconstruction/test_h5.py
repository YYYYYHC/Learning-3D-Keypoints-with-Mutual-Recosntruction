import time

import torch
import torch.nn as nn
from models.merger_net import Net
from datasets.dataset_h5 import all_h5
import matplotlib.pyplot as plt

def visualize3D(points, kpc):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xdata = points[:,0]
    ydata = points[:,1]
    zdata = points[:,2]
    ax.scatter3D(xdata, ydata, zdata, s=1)
    kxdata = kpc[:, 0]
    kydata = kpc[:, 1]
    kzdata = kpc[:, 2]
    ax.scatter3D(kxdata, kydata, kzdata, c='r', s=5)

if __name__ == '__main__':
    #load pretained net

    modeldir = ['models/check_points_h5/merger.pt.bak', 'models/check_points_h5/twosingle.pt']
    pretrained_net_dict = torch.load(modeldir[1])['model_state_dict']
    # print(type(pretrained_net_dict))

    pretrained_net = Net(2048, 10).to('cuda:2')
    pretrained_net = nn.DataParallel(pretrained_net, device_ids=[0,1,2]).module

    pretrained_net.load_state_dict(state_dict=pretrained_net_dict)

    # print(pretrained_net)

    #load test data
    x_test, xl_test = all_h5('./data/point_cloud/train', True, True, subclasses=(14,), sample=None)
    print(x_test.shape)
    #get keypoints
    batch_num = 8
    for i in range(x_test.shape[0] // batch_num):
        idx = slice(i * batch_num, (i+1)*batch_num)
        batch_x_test = torch.tensor(x_test[idx],device='cuda:2')
        with torch.no_grad():
            RPCD, KPCD, KPA, LF, MA = pretrained_net(batch_x_test)
            if(i==6):
                print(batch_x_test[0])
                print(KPCD[0])
                print(KPA[0])
                visualize3D(batch_x_test[0].cpu(), KPCD[0].cpu())
                print("len:{},{},{},{},{},{}".format(batch_x_test.shape, len(RPCD[0]), len(KPCD), len(KPA), len(LF[0]), len(MA[0])))
                print('*'*15)
                break

    #evaluate the keypoints