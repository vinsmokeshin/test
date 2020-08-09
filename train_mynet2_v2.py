import my_network2 as vm3d
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import numpy as np
import yaml
import argparse
import os

from tqdm import tqdm
import gc
import time
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import nibabel as nib
import random
import loss
from datetime import datetime

class VoxelMorph():
    """
    VoxelMorph Class is a higher level interface for both 2D and 3D
    Voxelmorph classes. It makes training easier and is scalable.
    """

    def __init__(self, config, use_gpu=False):
        self.dims = 3
        self.vm = vm3d
        self.voxelmorph = vm3d.VoxelMorph3d(config['in_channel'], 'bilinear', use_gpu)
        if use_gpu:
            self.voxelmorph = self.voxelmorph.cuda()
        # self.optimizer_c = optim.Adam(
        #     [{'params': self.voxelmorph.feature1.parameters()}, {'params': self.voxelmorph.feature2.parameters()},
        #      {'params': self.voxelmorph.feature3.parameters()}, {'params': self.voxelmorph.feature4.parameters()},
        #      {'params': self.voxelmorph.fusion_3.parameters()}, {'params': self.voxelmorph.fusion_a.parameters()},
        #      {'params': self.voxelmorph.deconv_4.parameters()}, {'params': self.voxelmorph.deconv_3.parameters()},
        #      {'params': self.voxelmorph.deconv_a.parameters()}, {'params': self.voxelmorph.conv.parameters()}], lr=1e-5)
        # self.optimizer_f = optim.Adam(
        #     [{'params': self.voxelmorph.feature1.parameters(), 'lr': 1e-6}, {'params': self.voxelmorph.feature2.parameters(),'lr': 5e-6},
        #      {'params': self.voxelmorph.conv2.parameters()}, {'params': self.voxelmorph.fusion_0.parameters()},
        #      {'params': self.voxelmorph.fusion_1.parameters()}, {'params': self.voxelmorph.fusion_2.parameters()},
        #      {'params': self.voxelmorph.deconv_0.parameters()}, {'params': self.voxelmorph.deconv_1.parameters()},
        #      {'params': self.voxelmorph.deconv_2.parameters()}], lr=1e-5)
        if config['Train']['optim'] == 'Adam':
            self.optimizer = optim.Adam(
                self.voxelmorph.parameters(), lr=config['Train']['lr'])
        self.device =  use_gpu
        self.criterian = nn.MSELoss(reduction='mean')
        # self.curvature = self.vm.curvature(use_gpu)
        # self.mmse = loss.MassMSE()
        self.sacd = loss.SACD()
        self.smooth = loss.GradientLoss()
        #self.antifold = loss.JacobAntiFoldingLoss()
        self.GCC = loss.GCCLoss(use_gpu=use_gpu)
        if config['Train']['pretrained'] is not None:
            self.load_model(config['Train']['pretrained'])
        self.a1 = config['Train']['a1']
        self.a2 = config['Train']['a2']
        self.b1 = config['Train']['b1']
        self.b2 = config['Train']['b2']
        self.b3 = config['Train']['b3']
        self.b4 = config['Train']['b4']

    def check_dims(self, x):
        try:
            if x.shape[1:] == self.dims:
                return
            else:
                raise TypeError
        except TypeError as e:
            print("Invalid Dimension Error. The supposed dimension is ",
                  self.dims, "But the dimension of the input is ", x.shape[1:])

    def train_model(self, batch_moving, batch_fixed, lamda=0.1):
        self.optimizer.zero_grad()
        if self.device is True:
            batch_fixed, batch_moving = batch_fixed.cuda(), batch_moving.cuda()
        registered_coarse, dvf_coarse = self.voxelmorph.coarse_path(batch_moving, batch_fixed)
        # ncc = self.vm.cross_correlation_loss(registered_image[:,:1,...],batch_fixed[:,:1,...])
        mse = self.criterian(registered_coarse[:, 3:8, ...], batch_fixed[:, 3:8, ...])
        sm_loss = self.smooth(dvf_coarse)
        ncc =(self.sacd(registered_coarse[:, 2:3, ...], batch_fixed[:, 9:10, ...]) +
                          self.sacd(batch_fixed[:, 2:3, ...], registered_coarse[:, 9:10, ...]))
        loss = self.a2*sm_loss + self.a1* mse
        # self.optimizer.zero_grad()
        loss.backward()
        if lamda != 0:
            dvf_coarse = dvf_coarse.detach()
            registered = registered_coarse[:, 0:3., ...].detach()
            registered_fine, deformation_fine = self.voxelmorph.fine_path(batch_moving, batch_fixed, registered, dvf_coarse)
            phi, theta = self.GCC(registered_fine[:, 8:9, ...], batch_fixed[:, 8:9, ...])
            error = (registered_fine[:, 0, ...] - batch_fixed[:, 0, ...]) ** 2 + phi + theta
            ncc = self.region(registered_fine[:, 3:8, ...], batch_fixed[:, 3:8, ...], error)
            sm_loss = self.smooth(deformation_fine)
            loss2 = ncc+3*sm_loss
            loss2.backward()
        else:
            dvf_coarse = dvf_coarse.detach()
            registered = registered_coarse[:, 0:3, ...].detach()
            registered_fine, deformation_fine = self.voxelmorph.fine_path(batch_moving, batch_fixed, registered, dvf_coarse)
            phi, theta = self.GCC(registered_fine[:, 8:9, ...], batch_fixed[:, 8:9, ...])
            #error = (registered_fine[:, 0, ...] - batch_fixed[:, 0, ...]) ** 2 + phi + theta
            # ncc = self.region(registered_fine[:, 3:8, ...], batch_fixed[:, 3:8, ...], error)
            mse = self.criterian(registered_fine[:, [0,1], ...], batch_fixed[:, [0,1], ...])
            if self.b2 != 0:
                phi = torch.mean(torch.mean(phi,dim=[1,2,3]))
                theta = torch.mean(torch.mean(theta, dim=[1,2,3]))
            else:
                phi=0
                theta=0
            ncc =(self.sacd(registered_fine[:, 2:3, ...], batch_fixed[:, 9:10, ...]) +
                          self.sacd(batch_fixed[:, 2:3, ...], registered_fine[:, 9:10, ...]))
            sm_loss = self.smooth(deformation_fine)
            loss2 = self.b1*mse+self.b2*(phi+theta)+ self.b3*ncc + self.b4*sm_loss
            loss2.backward()
        self.optimizer.step()
        return loss, loss2, ncc, sm_loss

    def get_test_loss(self, batch_moving, batch_fixed, lamda=0.1):
        with torch.no_grad():
            batch_fixed, batch_moving = batch_fixed.cuda(), batch_moving.cuda()
            registered_image, deformation_matrix = self.voxelmorph(batch_moving, batch_fixed, False)
            # val_loss = self.vm.vox_morph_loss(
            #    registered_image, batch_fixed,deformation_matrix, n, lamda, use_gpu)
            # ncc = self.vm.cross_correlation_loss(registered_image[:,:1,...],batch_fixed[:,:1,...])
            # ncc = self.criterian(registered_image[:, :1, ...], batch_fixed[:, :1, ...])
            mse = self.criterian(registered_image[:, [0,2], ...], batch_fixed[:, [0,2], ...])
            if self.b2 != 0:
                phi, theta = self.GCC(registered_image[:, 8:9, ...], batch_fixed[:, 8:9, ...])
                phi = torch.mean(torch.mean(phi,dim=[1,2,3]))
                theta = torch.mean(torch.mean(theta, dim=[1,2,3]))
            else:
                phi=0
                theta=0
            if self.b3 != 0:
                ncc =(self.sacd(registered_image[:, 2:3, ...], batch_fixed[:, 9:10, ...]) +
                              self.sacd(batch_fixed[:, 2:3, ...], registered_image[:, 9:10, ...]))
            else:
                ncc=0
            val_loss = self.criterian(batch_fixed[:, 2:8, ...], registered_image[:, 2:8, ...])
            sm_loss = self.smooth(deformation_matrix)
            loss = self.b1*mse+self.b2*(phi+theta)+ self.b3*ncc + self.b4*sm_loss

        return val_loss, loss, sm_loss, registered_image.data

    def load_model(self, checkpoint_path):
        self.voxelmorph.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint_path).items()})


def LabelToOnehot(img, label_num):
    onehot = np.zeros((label_num - 1, img.shape[0], img.shape[1], img.shape[2]))
    for i in range(1, label_num):
        onehot[i - 1, :, :, :] = img == i
    return onehot


def Onehot2Label(img):
    onehot = np.zeros((2, img.shape[0], img.shape[1], img.shape[2]))
    for i in range(img.shape[4] - 1):
        onehot[0] += img[:, :, :, 0, i] * (i + 1)
    onehot[1] = img[:, :, :, 0, 5]
    return onehot


class Dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, normalize=True):
        'Initialization'
        self.list_IDs = []
        self.image = {}
        self.mask = {}
        self.airway = {}
        self.vesselness = {}
        self.dst = {}
        self.flag = {}
        self.list_tgt = list_IDs[1::2]
        self.list_src = list_IDs[::2]
        for item in list_IDs:
            self.image[item] = nib.load(item)
            try:
                self.mask[item] = nib.load(item.replace('_d.nii.gz', '_lobule_d.nii.gz'))
                self.airway[item] = nib.load(item.replace('_d.nii.gz', '_airway_d.nii.gz'))
                self.dst[item] = nib.load(item.replace('_d.nii.gz', '_dst_d.nii.gz'))
                basename = os.path.basename(item)
                self.vesselness[item] = nib.load(item.replace(basename, 'frangi_d.nii.gz'))
                self.flag[item] = 1
            except:
                self.mask[item] = nib.load(item.replace('image_d.nii.gz', 'label_d.nii.gz'))
                self.vesselness[item] = nib.load(item.replace('image_d.nii.gz', 'frangi_d.nii.gz'))
                self.dst[item] = nib.load(item.replace('image_d.nii.gz', 'airway_dst_d.nii.gz'))
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
            fixed_image = (fixed_image + 1000) / 160.0
            moving_image = torch.clamp(moving_image, -1000, 600)
            moving_image = (moving_image + 1000) / 160.0

        fixed_vessel = self.vesselness[fixedfile].get_data()
        fixed_vessel = torch.Tensor(fixed_vessel)[np.newaxis, ...]
        fixed_dst = self.dst[fixedfile].get_data()
        fixed_dst = torch.Tensor(fixed_dst)[np.newaxis, ...]
        if self.flag[fixedfile] == 0:
            fixed_onehot = self.mask[fixedfile].get_data()
            fixed_mask = Onehot2Label(fixed_onehot)
            fixed_mask = torch.Tensor(fixed_mask)
            fixed_vessel = torch.where(fixed_mask[0:1,...]>0, fixed_vessel, fixed_mask[0:1,...])
            fixed_onehot = torch.Tensor(fixed_onehot)[:, :, :, 0, 0:5].permute(3, 0, 1, 2)
            fixed_image = torch.cat((fixed_image, fixed_mask, fixed_onehot, fixed_vessel, fixed_dst), 0)
        else:
            fixed_mask = self.mask[fixedfile].get_data().astype(np.float32)
            fixed_onehot = torch.Tensor(LabelToOnehot(fixed_mask, 6))
            fixed_mask = torch.Tensor(fixed_mask)[np.newaxis, ...]
            fixed_vessel = torch.where(fixed_mask>0, fixed_vessel, fixed_mask)
            fixed_airway = self.airway[fixedfile].get_data()
            fixed_airway = torch.Tensor(fixed_airway)[np.newaxis, ...]
            fixed_image = torch.cat((fixed_image, fixed_mask, fixed_airway, fixed_onehot, fixed_vessel, fixed_dst), 0)

        moving_vessel = self.vesselness[movfile].get_data()[np.newaxis, ...]
        moving_vessel = torch.Tensor(moving_vessel)
        moving_dst = self.dst[movfile].get_data()
        moving_dst = torch.Tensor(moving_dst)[np.newaxis, ...]
        if self.flag[movfile] == 0:

            # moving_mask = LabelToOnehot(moving_mask, 4)
            moving_onehot = self.mask[movfile].get_data()
            moving_mask = Onehot2Label(moving_onehot)
            moving_mask = torch.Tensor(moving_mask)
            moving_vessel = torch.where(moving_mask[0:1,...]>0, moving_vessel, moving_mask[0:1,...])
            moving_onehot = torch.Tensor(moving_onehot)[:, :, :, 0, 0:5].permute(3, 0, 1, 2)
            moving_image = torch.cat((moving_image, moving_mask, moving_onehot, moving_vessel, moving_dst), 0)
        else:
            moving_mask = self.mask[movfile].get_data().astype(np.float32)
            moving_onehot = torch.Tensor(LabelToOnehot(moving_mask, 6))
            moving_mask = torch.Tensor(moving_mask)[np.newaxis, ...]
            moving_vessel = torch.where(moving_mask>0, moving_vessel, moving_mask)
            moving_airway = self.airway[movfile].get_data()
            moving_airway = torch.Tensor(moving_airway)[np.newaxis, ...]
            moving_image = torch.cat(
                (moving_image, moving_mask, moving_airway, moving_onehot, moving_vessel, moving_dst), 0)

        # fixedfile = fixedfile.split('/')[-2]
        # print(time.time())
        return fixedfile, movfile, fixed_image, moving_image


def save_paras(cfg, path):
	with open(os.path.join(path,'config.yaml'),"w",encoding="utf-8") as f:
		yaml.dump(cfg,f)


def save_checkpoints(model, step, dirname, isStep=True):
    # Recommand: save and load only the model parameters
    if isStep:
        step = str(step)
    filename = 'network_' + step + '.pth'
    torch.save(model.state_dict(), os.path.join(dirname, filename))
    print("===> ===> ===> Save checkpoint {} to {}".format(step, filename))
    
    
def adjust_learning_rate(optimizer, epoch, init_lr, decay_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (decay_rate ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def arg_parse():
    parser = argparse.ArgumentParser(description='registration')
    parser.add_argument('-cfg', '--config', default='config/config.yaml', type=str, help='load the config file')
    args = parser.parse_args()
    return args


def main():
    '''
    In this I'll take example of FIRE: Fundus Image Registration Dataset
    to demostrate the working of the API.
    '''
    args = arg_parse()
    print(args.config)
    config = yaml.load(open(args.config))
    cp_path=config['Train']['checkpoint_path']
    if not os.path.exists(cp_path):
        os.makedirs(cp_path)
    save_paras(config,cp_path)
    gpus = ','.join(str(i) for i in config['GPUs'])
    print(gpus)
    #os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpus
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    writer = SummaryWriter("/home/lynn/code/semantic_reg/train_log/" + config['train_log'] + TIMESTAMP)
    use_gpu = torch.cuda.is_available()
    vm = VoxelMorph(
        config, use_gpu=use_gpu)  # Object of the higher level class
    params = {'batch_size': config['Train']['batchsize'],
              'shuffle': config['Train']['shuffle'],
              'num_workers': config['Train']['workers'],
              # 'worker_init_fn': np.random.seed(42)
              }

    # to change
    max_epochs = config['Train']['max_epoch']
    # filename = [line.rstrip("\n") for line in open('./datalist/followup.txt')]
    partition = {}
    partition['train'] = [line.rstrip("\n") for line in open(config['Train']['train_list'])]
    partition['validation'] = [line.rstrip("\n") for line in open(config['Train']['val_list'])]
    print(len(partition['train']))
    print(len(partition['validation']))

    # Generators
    training_set = Dataset(partition['train'])
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(partition['validation'])
    validation_generator = data.DataLoader(validation_set, **params)
    lamba = 0
    min_mse = 10
    min_mse_label = 10
    # Loop over epochs
    for epoch in range(0, max_epochs + 1):
        start_time = time.time()
        train_loss = 0
        train_dice_score = 0
        train_cur_score = 0
        train_sm_score = 0
        val_loss = 0
        val_dice_score = 0
        val_cur_score = 0
        val_sm_score = 0
        for fixed, _, batch_fixed, batch_moving in training_generator:
            loss, dice, cur, sm = vm.train_model(batch_moving, batch_fixed, lamda=lamba)
            train_dice_score += dice.data.item()
            train_cur_score += cur.data.item()
            train_sm_score += sm.data.item()
            train_loss += loss.data.item()
        print('[', "{0:.2f}".format((time.time() - start_time) / 60), 'mins]', 'epochs:', epoch, ' loss:',
              train_loss * params['batch_size'] / len(training_set), ' Dice:', train_dice_score *
              params['batch_size'] / len(training_set), ' cur:',
              train_cur_score * params['batch_size'] / len(training_set), ' sm:',
              train_sm_score * params['batch_size'] / len(training_set))  # train_dice_score.data
        # Testing time
        start_time = time.time()
        writer.add_scalar("train/train_loss", train_loss * params['batch_size'] / len(training_set), epoch)
        writer.add_scalar("train/mse_loss", train_dice_score * params['batch_size'] / len(training_set), epoch)
        writer.add_scalar("train/ncc_loss", train_cur_score * params['batch_size'] / len(training_set), epoch)
        writer.add_scalar("train/sm_loss", train_sm_score * params['batch_size'] / len(training_set), epoch)
        if epoch % config['Train']['lr_decay_freq'] == 0:    
            step = epoch//config['Train']['lr_decay_freq']
            adjust_learning_rate(vm.optimizer, step, config['Train']['lr'], config['Train']['lr_decay_rate'])
        #if epoch == 30:
        #    lamba = 1
        for fix, mov, batch_fixed, batch_moving in validation_generator:
            # Transfer to GPU
            dice, cur, sm, registered_image = vm.get_test_loss(batch_moving, batch_fixed)
            b = batch_fixed.shape[0]
            val_dice_score += dice.item()*b
            val_cur_score += cur.item()*b
            val_sm_score += sm.item()*b
            # to change
            if epoch % 20 == 0:
                for i in range(b):
                    _, _, x, y, z = batch_fixed.shape
                    moved = batch_moving[i, 0:1, :, :, z // 2].cpu().numpy()
                    flowed = registered_image[i, 0:1, :, :, z // 2].cpu().numpy()
                    fixed = batch_fixed[i, 0:1, :, :, z // 2].cpu().numpy()
                    writer.add_image("val/" + fix[i] + "_" + mov[i] + "_0", fixed/10, epoch, dataformats='CHW')
                    writer.add_image("val/" + fix[i] + "_" + mov[i] + "_2", moved/10, epoch, dataformats='CHW')
                    writer.add_image("val/" + fix[i] + "_" + mov[i] + "_1", flowed/10, epoch, dataformats='CHW')
        val_dice_score = val_dice_score / len(validation_set)
        val_cur_score = val_cur_score / len(validation_set)
        val_sm_score = val_sm_score / len(validation_set)
        print('[', "{0:.2f}".format((time.time() - start_time) / 60), 'mins]', 'epochs:', epoch, ' Dice:', val_dice_score, ' cur:', val_cur_score, ' sm:',val_sm_score)  # train_dice_score.data
        writer.add_scalar("val/mse_loss", val_dice_score, epoch)
        writer.add_scalar("val/ncc_loss", val_cur_score, epoch)
        writer.add_scalar("val/sm_loss", val_sm_score, epoch)
        if val_dice_score < min_mse_label:
            save_checkpoints(vm.voxelmorph,'label',cp_path,False)
            min_mse_label = val_dice_score
        if val_cur_score < min_mse:
            save_checkpoints(vm.voxelmorph,epoch, cp_path)
            min_mse = val_cur_score
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"]="3"
    main()
