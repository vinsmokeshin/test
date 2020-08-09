import voxelmorph2d as vm2d
import my_network2 as vm3d
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
from skimage.transform import resize
import multiprocessing as mp
from tqdm import tqdm
import gc
import time
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
import nibabel as nib
import loss
import random
import argparse
use_gpu = torch.cuda.is_available()

#from luna16 import LUNA16
def arg_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('-m','--model',default='/home/lynn/code/VoxelMorph-PyTorch/mynetwork2_v/network_263.pth')
    parser.add_argument('-c','--channel',default=3)
    args = parser.parse_args()
    return args

arg = arg_parse()

class VoxelMorph():
    """
    VoxelMorph Class is a higher level interface for both 2D and 3D
    Voxelmorph classes. It makes training easier and is scalable.
    """

    def __init__(self, input_channel,model_path, is_2d=False, use_gpu=False):
        self.dims = input_channel
        if is_2d:
            self.vm = vm2d
            self.voxelmorph = vm2d.VoxelMorph2d(input_channel, use_gpu)
        else:
            self.vm = vm3d
            self.voxelmorph = vm3d.VoxelMorph3d(input_channel, 'nearest', use_gpu)
            #self.voxelmorph = vm3d.VoxelMorph3d(input_channel, 'bilinear', use_gpu)
        if use_gpu:
            self.voxelmorph = self.voxelmorph.cuda()
        #self.load_model('/home/lynn/code/VoxelMorph-PyTorch/mynetwork2_v/network_4.pth')
        self.load_model(model_path)
        #self.load_model('/home/lynn/code/VoxelMorph-PyTorch/mynetwork2_dirlab/network_3.pth')
        self.device = torch.device("cuda:0" if use_gpu else "cpu")
        self.criterian = nn.MSELoss(reduction='mean')
        self.jaco = loss.JacobScore()
        self.smooth = loss.GradientLoss()

    def check_dims(self, x):
        try:
            if x.shape[1:] == self.dims:
                return
            else:
                raise TypeError
        except TypeError as e:
            print("Invalid Dimension Error. The supposed dimension is ",
                  self.dims, "But the dimension of the input is ", x.shape[1:])

    def forward(self, x):
        # self.check_dims(x)
        return voxelmorph(x)

    def get_test(self, batch_moving, batch_fixed):
        with torch.set_grad_enabled(False):
            batch_fixed, batch_moving = batch_fixed.to(
                self.device), batch_moving.to(self.device)
            registered_image, deformation_matrix = self.voxelmorph(batch_moving, batch_fixed)
            registered = self.voxelmorph.spatial_transform(batch_moving[:,8:9,...], deformation_matrix, 'bilinear')
            #ncc = self.criterian(registered, batch_fixed[:, 8:9, ...])
            #registered_image = batch_moving
            #registered = batch_moving[:, 8:9,...]
            error = torch.abs(registered - batch_fixed[:, 8:9, ...])
            mask_fixed = (batch_fixed[:,1:2,...]>0).float()
            mask_warp = (registered_image[:,1:2,...]>0).float()
            mask = mask_fixed + mask_warp - mask_fixed * mask_warp
            #mask = mask_fixed * mask_warp
            #ncc = self.vm.cross_correlation_loss(registered, batch_fixed[:, 0:1, ...])
            ncc = torch.sum(error*mask)/mask.sum()
            
            #mask_fixed = torch.where(batch_fixed[:,1:2,...]>0,batch_fixed[:, 8:9, ...],batch_fixed[:,1:2,...])
            #mask_warp = torch.where(registered_image[:,1:2,...]>0, registered, registered_image[:,1:2,...])
            #ncc = self.criterian(mask_warp, mask_fixed)
            airway_dice = self.vm.dice_score(batch_fixed[:, 2:3, ...], registered_image[:, 2:3, ...])
            lobule_dice1 = self.vm.dice_score(batch_fixed[:, 3:4, ...], registered_image[:, 3:4, ...])
            lobule_dice2 = self.vm.dice_score(batch_fixed[:, 4:5, ...], registered_image[:, 4:5, ...])
            lobule_dice3 = self.vm.dice_score(batch_fixed[:, 5:6, ...], registered_image[:, 5:6, ...])
            lobule_dice4 = self.vm.dice_score(batch_fixed[:, 6:7, ...], registered_image[:, 6:7, ...])
            lobule_dice5 = self.vm.dice_score(batch_fixed[:, 7:8, ...], registered_image[:, 7:8, ...])
            # ncc = self.criterian(batch_moving[:, :1, ...], batch_fixed[:, :1, ...])
            # airway_dice = self.vm.dice_score(batch_fixed[:, 2:3, ...], batch_moving[:, 2:3, ...])
            # lobule_dice = self.vm.dice_score(batch_fixed[:, 3:8, ...], batch_moving[:, 3:8, ...])
            jaco = self.smooth(deformation_matrix)
            #jaco = torch.Tensor(0)
            # neg_num = torch.Tensor(0)
            return ncc, airway_dice, lobule_dice1, lobule_dice2, lobule_dice3, lobule_dice4, lobule_dice5, jaco

    def load_model(self, checkpoint_path):
        self.voxelmorph.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint_path).items()})


class Dataset(data.Dataset):
    def __init__(self, list_IDs, normalize=True):
        'Initialization'
        self.list_IDs = []
        self.image = {}
        self.mask = {}
        self.airway = {}
        self.vesselness = {}
        self.dst = {}
        self.flag = {}
        self.list_tgt = list_IDs[::2]
        self.list_src = list_IDs[1::2]
        for item in list_IDs:
            self.image[item] = nib.load(item)
            try:
                self.mask[item] = nib.load(item.replace('_d.nii.gz', '_lobule_d.nii.gz'))
                self.airway[item] = nib.load(item.replace('_d.nii.gz', '_airway_d.nii.gz'))
                self.flag[item] = 1
            except:
                self.mask[item] = nib.load(item.replace('image_d.nii.gz', 'label_d.nii.gz'))
                self.flag[item] = 0
        self.norm = normalize

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_tgt)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # print(time.time())
        fixedfile = self.list_tgt[index]
        movfile = self.list_src[index]
        # index1 = np.random.randint(0,len(self.list_IDs))

        # Load data and get label
        fixed_image = self.image[fixedfile].get_data()
        H, W, D = fixed_image.shape
        fixed_image = fixed_image[np.newaxis, ...]
        # print(fixed_image.shape)
        fixed_image = torch.Tensor(fixed_image)
        moving_image = self.image[movfile].get_data()
        moving_image = moving_image[np.newaxis, ...]
        moving_image = torch.Tensor(moving_image)
        if self.norm:
            fixed_image = torch.clamp(fixed_image, -1000, 600)
            fixed_lung = (fixed_image + 1000) / 160.0
            moving_image = torch.clamp(moving_image, -1000, 600)
            moving_lung = (moving_image + 1000) / 160.0

        if self.flag[fixedfile] == 0:
            fixed_onehot = self.mask[fixedfile].get_data()
            fixed_mask = Onehot2Label(fixed_onehot)
            fixed_mask = torch.Tensor(fixed_mask)
            fixed_onehot = torch.Tensor(fixed_onehot)[:, :, :, 0, 0:5].permute(3, 0, 1, 2)
            fixed_image = torch.cat((fixed_lung, fixed_mask, fixed_onehot, fixed_image), 0)
        else:
            fixed_mask = self.mask[fixedfile].get_data().astype(np.float32)
            fixed_onehot = torch.Tensor(LabelToOnehot(fixed_mask, 6))
            fixed_mask = torch.Tensor(fixed_mask)[np.newaxis, ...]
            fixed_airway = self.airway[fixedfile].get_data()
            fixed_airway = torch.Tensor(fixed_airway)[np.newaxis, ...]
            fixed_image = torch.cat((fixed_lung, fixed_mask, fixed_airway, fixed_onehot, fixed_image), 0)

        if self.flag[movfile] == 0:

            # moving_mask = LabelToOnehot(moving_mask, 4)
            moving_onehot = self.mask[movfile].get_data()
            moving_mask = Onehot2Label(moving_onehot)
            moving_mask = torch.Tensor(moving_mask)
            moving_onehot = torch.Tensor(moving_onehot)[:, :, :, 0, 0:5].permute(3, 0, 1, 2)
            moving_image = torch.cat((moving_lung, moving_mask, moving_onehot, moving_image), 0)
        else:
            moving_mask = self.mask[movfile].get_data().astype(np.float32)
            moving_onehot = torch.Tensor(LabelToOnehot(moving_mask, 6))
            moving_mask = torch.Tensor(moving_mask)[np.newaxis, ...]
            moving_airway = self.airway[movfile].get_data()
            moving_airway = torch.Tensor(moving_airway)[np.newaxis, ...]
            moving_image = torch.cat(
                (moving_lung, moving_mask, moving_airway, moving_onehot, moving_image), 0)

        # fixedfile = fixedfile.split('/')[-2]
        # print(time.time())
        return fixedfile, movfile, fixed_image, moving_image


def LabelToOnehot(img, label_num):
    onehot = np.zeros((label_num - 1, img.shape[0], img.shape[1], img.shape[2]))
    for i in range(1, label_num):
        onehot[i - 1, :, :, :] = img == i
    return onehot


def Onehot2Label(img):
    onehot = np.zeros((2, img.shape[0], img.shape[1], img.shape[2]))
    for i in range(img.shape[4] - 1):
        onehot[0] += img[:, :, :, 0, i] * (i + 1)  # *0.2
    onehot[1] = img[:, :, :, 0, 5]
    return onehot


def main():
    '''
    In this I'll take example of FIRE: Fundus Image Registration Dataset
    to demostrate the working of the API.
    '''
    vm = VoxelMorph(
        arg.channel,arg.model, is_2d=False, use_gpu=use_gpu)  # Object of the higher level class
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1,
              # 'worker_init_fn': np.random.seed(42)
              }
    #filename = [line.rstrip("\n") for line in open('./datalist/dirlab.txt')]
    filename = [line.rstrip("\n") for line in open('./datalist/list2.txt')]

    validation_set = Dataset(filename)
    validation_generator = data.DataLoader(validation_set, **params)
    val_mse = []
    val_airway = []
    val_lobule1 = []
    val_lobule2 = []
    val_lobule3 = []
    val_lobule4 = []
    val_lobule5 = []
    val_jaco = []
    val_num = []
    val_sm_score = 0
    lobule_min = 1
    lobule_max = 0
    airway_min = 1
    airway_max = 0
    for fix, mov, batch_fixed, batch_moving in validation_generator:
        # Transfer to GPU
        # print(fix)
        mse, airway, lobule1, lobule2, lobule3, lobule4, lobule5, jaco = vm.get_test(batch_moving, batch_fixed)
        val_mse.append(mse.data.cpu().numpy())
        val_airway.append(airway.data.cpu().numpy())
        val_lobule1.append(lobule1.data.cpu().numpy())
        val_lobule2.append(lobule2.data.cpu().numpy())
        val_lobule3.append(lobule3.data.cpu().numpy())
        val_lobule4.append(lobule4.data.cpu().numpy())
        val_lobule5.append(lobule5.data.cpu().numpy())
        val_jaco.append(jaco.data.cpu().numpy())
        #val_num.append(num.data.cpu().numpy())
    print(val_mse)
    print(val_airway)
    print(val_lobule1)
    print(val_lobule2)
    print(val_lobule3)
    print(val_lobule4)
    print(val_lobule5)
    val_mse = np.array(val_mse)
    val_airway = np.array(val_airway)
    val_lobule1 = np.array(val_lobule1)
    val_lobule2 = np.array(val_lobule2)
    val_lobule3 = np.array(val_lobule3)
    val_lobule4 = np.array(val_lobule4)
    val_lobule5 = np.array(val_lobule5)
    val_jaco = np.array(val_jaco)
    #val_num = np.array(val_num)
    print('mse:', val_mse.mean(), val_mse.std(),
          ' airway:', val_airway.mean(), val_airway.std(), ' lobule1:', val_lobule1.mean(), val_lobule1.std(), 
          ' lobule2:', val_lobule2.mean(), val_lobule2.std(), ' lobule3:', val_lobule3.mean(), val_lobule3.std(), 
          ' lobule4:', val_lobule4.mean(), val_lobule4.std(), ' lobule5:', val_lobule5.mean(), val_lobule5.std(), 
          ' jaco:', val_jaco.mean(), val_jaco.std(), ' num:')#,val_num.mean(),
          #val_jaco.std())  # train_dice_score.data


if __name__ == "__main__":
    main()
