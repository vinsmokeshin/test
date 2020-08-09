import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from network import unet_core, unet_nopadding


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dim=3):
        """
        + Instantiate modules: conv-relu-norm
        + Assign them as member variables
        """
        super(conv_block, self).__init__()
        conv_type = getattr(nn, 'Conv%dd' % dim)
        self.conv = conv_type(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.leakyrelu = nn.ReLU()
        # with learnable parameters
        # self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        return self.leakyrelu(self.conv(x))


class fusion_conv(nn.Module):
    def __init__(self, in_channels, out_channels, dim=3):
        """
        + Instantiate modules: conv-relu-norm
        + Assign them as member variables
        """
        super(fusion_conv, self).__init__()
        conv_type = getattr(nn, 'Conv%dd' % dim)
        self.conv = conv_type(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # with learnable parameters
        # self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        return self.conv(x)


class SpatialTransformation(nn.Module):
    def meshgrid(self, n, height, width, depth):
        '''x_t = torch.matmul(torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0, width-1, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0, height-1, height), 1), torch.ones([1, width]))
        z_t = torch.matmul(torch.unsqueeze(torch.linspace(0, depth - 1, depth), 1), torch.ones([1, height]))
        z_t = z_t[...,np.newaxis]

        grid[:,:,:,:,2] = x_t.expand([n,depth,height, width])
        grid[:,:,:,:,1] = y_t.expand([n,depth,height, width])
        z_t = z_t.expand([depth,height, width])
        grid[:,:,:,:,0] = z_t.expand([n,depth,height, width])
        if self.use_gpu==True:
            grid = grid.cuda()'''

        # grid_h, grid_w, grid_d = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1,1,width),torch.linspace(-1, 1, depth))
        grid_h, grid_w, grid_d = torch.meshgrid(torch.linspace(0, height - 1, height),
                                                torch.linspace(0, width - 1, width),
                                                torch.linspace(0, depth - 1, depth))
        grid = torch.stack((grid_d, grid_w, grid_h), 3)
        grid = grid.expand([n, height, width, depth, 3])
        if self.use_gpu == True:
            grid = (grid.cuda()).contiguous()
        return grid

    def __init__(self, use_gpu=False, full_size=True):
        self.use_gpu = use_gpu
        self.full_size = full_size
        super(SpatialTransformation, self).__init__()
        # self.grid = self.meshgrid(N, H, W, D)

    def forward(self, moving_image, deformation_matrix, mode, grid=None):
        # print(grid)
        N, C, H, W, D = moving_image.shape
        if grid is None:
            grid = self.meshgrid(N, H, W, D)
        if self.full_size:
            deformation_matrix = deformation_matrix.permute(0, 2, 3, 4, 1)
            grid = grid + deformation_matrix
        # grid = grid.contiguous()
        grid[..., 0] = grid[..., 0] / (D - 1) * 2 - 1
        grid[..., 1] = grid[..., 1] / (W - 1) * 2 - 1
        grid[..., 2] = grid[..., 2] / (H - 1) * 2 - 1
        # flag = torch.min((grid > 0)-(grid > 1),4)[0][:,np.newaxis,...]
        # return F.grid_sample(moving_image,grid,padding_mode = 'zeros'), flag
        return F.grid_sample(moving_image, grid, mode=mode, padding_mode='zeros')


class VoxelMorph3d(nn.Module):
    def __init__(self, in_channels, mode='nearest', use_gpu=False):
        super(VoxelMorph3d, self).__init__()
        # self.unet = unet_nopadding(3,[16,16,32,32], src_feats=1)
        # self.conv = nn.Conv3d(16,3,3,padding=1)
        # self.unet = unet_core(3, [32, 64, 128, 256], [128, 64, 32, 32, 32, 16, 16], src_feats=in_channels)
        self.feature1 = conv_block(in_channels, 32, stride=2, dim=3)
        self.feature2 = conv_block(32, 64, stride=2, dim=3)
        self.feature3 = conv_block(64, 64, stride=2, dim=3)
        self.feature4 = conv_block(64, 125, stride=2, dim=3)
        self.fusion_0 = conv_block(in_channels * 2, 16, stride=1, dim=3)
        self.fusion_1 = nn.Conv3d(32 * 2, 32, kernel_size=1)
        self.fusion_2 = nn.Conv3d(64 * 2, 64, kernel_size=1)
        self.fusion_3 = nn.Conv3d(64 * 2, 64, kernel_size=1)
        self.fusion_a = nn.Conv3d(64 * 2, 64, kernel_size=1)
        self.fusion_a_1 = nn.Conv3d(32 * 2, 32, kernel_size=1)
        self.deconv_a_1 = conv_block(64 + 32, 32, stride=1, dim=3)
        self.deconv_a = conv_block(128 + 64, 64, stride=1, dim=3)
        self.deconv_4 = conv_block(125 * 2, 128, stride=1, dim=3)
        self.deconv_3 = conv_block(128 + 64, 128, stride=1, dim=3)
        self.deconv_2 = conv_block(64 + 64, 64, stride=1, dim=3)
        self.deconv_1 = conv_block(64 + 64 + 32, 64, stride=1, dim=3)
        self.deconv_0 = conv_block(64 + 16, 32, stride=1, dim=3)
        self.conv = nn.Conv3d(32, 3, 3, padding=1)
        self.conv2 = nn.Conv3d(32, 3, 3, padding=1)
        self.spatial_transform = SpatialTransformation(use_gpu, full_size=True)
        self.int_mode = mode
        self.channel = in_channels
        '''if use_gpu:
            self.unet = self.unet.cuda()
            self.conv = self.conv.cuda()
            self.spatial_transform = self.spatial_transform.cuda()'''

    def coarse_path(self, moving_image, fixed_image):
        fixed = fixed_image[:, :self.channel, ...]
        mov1 = self.feature1(moving_image[:, :self.channel, ...])
        mov2 = self.feature2(mov1)
        mov3 = self.feature3(mov2)
        mov4 = self.feature4(mov3)
        fix1 = self.feature1(fixed)
        fix2 = self.feature2(fix1)
        fix3 = self.feature3(fix2)
        fix4 = self.feature4(fix3)
        deconv = F.interpolate(self.deconv_4(torch.cat((fix4, mov4), 1)), scale_factor=2, mode='trilinear',
                               align_corners=True)
        deconv = F.interpolate(self.deconv_3(torch.cat((deconv, self.fusion_3(torch.cat((fix3, mov3), 1))), 1)),
                               scale_factor=2, mode='trilinear', align_corners=True)
        deconv = F.interpolate(self.deconv_a(torch.cat((deconv, self.fusion_a(torch.cat((fix2, mov2), 1))), 1)),
                               scale_factor=2, mode='trilinear', align_corners=True)
        deconv = self.deconv_a_1(torch.cat((deconv, self.fusion_a_1(torch.cat((fix1, mov1), 1))), 1))
        dvf_coarse = F.interpolate(self.conv(deconv), scale_factor=2, mode='trilinear',
                                   align_corners=True)
        registered_coarse = self.spatial_transform(moving_image, dvf_coarse, self.int_mode)
        return registered_coarse, dvf_coarse

    def fine_path(self, moving_image, fixed_image, registered_coarse, dvf_coarse):
        fixed = fixed_image[:, :self.channel, ...]
        mov1 = self.feature1(registered_coarse)
        mov2_ = self.feature2(mov1)
        fix1 = self.feature1(fixed)
        fix2 = self.feature2(fix1)
        deconv = self.deconv_2(torch.cat((fix2, mov2_), 1))  # 64
        deconv = F.interpolate(torch.cat((deconv, self.fusion_2(torch.cat((fix2, mov2_), 1))), 1), scale_factor=2,
                               mode='trilinear', align_corners=True)  # 64 + 64
        deconv = F.interpolate(self.deconv_1(torch.cat((deconv, self.fusion_1(torch.cat((fix1, mov1), 1))), 1)),
                               scale_factor=2, mode='trilinear', align_corners=True)
        deconv = self.deconv_0(torch.cat((deconv, self.fusion_0(torch.cat((fixed, registered_coarse), 1))), 1))
        dvf_fine = self.conv2(deconv)
        grid = self.spatial_transform.meshgrid(fixed.shape[0], fixed.shape[2], fixed.shape[3],
                                               fixed.shape[4])
        # registered_fine = self.spatial_transform(registered_coarse, dvf_fine, self.int_mode)
        final_dvf = self.spatial_transform(dvf_coarse, dvf_fine, self.int_mode, grid) + dvf_fine
        registered_fine = self.spatial_transform(moving_image, final_dvf, self.int_mode, grid)
        del (grid)
        return registered_fine, final_dvf

    def forward(self, moving_image, fixed_image, train=False):
        registered_coarse, dvf_coarse = self.coarse_path(moving_image, fixed_image)
        moving = registered_coarse[:, :self.channel, ...]
        registered_fine, final_dvf = self.fine_path(moving_image, fixed_image, moving, dvf_coarse)
        if train:
            return registered_fine, final_dvf, registered_coarse, dvf_coarse
        return registered_fine, final_dvf


def cross_correlation(I, J, n, use_gpu=False):
    batch_size, channels, zdim, ydim, xdim = I.shape
    I2 = torch.mul(I, I)
    J2 = torch.mul(J, J)
    IJ = torch.mul(I, J)
    _filter = torch.ones((n, n, n))
    sum_filter = torch.zeros((channels, channels, n, n, n))
    for i in range(channels):
        sum_filter[i, i] = _filter
    # print(sum_filter)
    if use_gpu:
        sum_filter = sum_filter.cuda()
    I2_sum = torch.conv3d(I2, sum_filter, padding=n // 2, stride=(1, 1, 1))
    J2_sum = torch.conv3d(J2, sum_filter, padding=n // 2, stride=(1, 1, 1))
    IJ_sum = torch.conv3d(IJ, sum_filter, padding=n // 2, stride=(1, 1, 1))
    cross = IJ_sum * IJ_sum
    cc = cross / (J2_sum * I2_sum + 0.000001)
    # print(torch.mean(cc))
    return torch.mean(cc)


def smooothing_loss(y_pred):
    # y_pred=y_pred.permute(0,2,3,4,1)
    dz = y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]
    dy = y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]
    dx = y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]
    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    dz = torch.mul(dz, dz)
    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    # print(d/2.0)
    return d


def cross_correlation_loss(I, J, use_gpu=False, patch=True):
    batch_size, channels, zdim, ydim, xdim = I.shape
    I_mean = torch.mean(I, [2, 3, 4]).view(batch_size, channels, 1, 1, 1)
    J_mean = torch.mean(J, [2, 3, 4]).view(batch_size, channels, 1, 1, 1)
    _I = I - I_mean
    _J = J - J_mean
    up = torch.sum(_I * _J, [2, 3, 4]) ** 2
    down = torch.sum(_I ** 2, [2, 3, 4]) * torch.sum(_J ** 2, [2, 3, 4])
    cc = up / down
    return 1 - torch.mean(cc)


def dice_score(y_conv, y_true):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    intersection = torch.sum(y_conv * y_true, [2, 3, 4])
    # `dim = 0` for Tensor result
    union = torch.sum(y_conv * y_conv, [2, 3, 4]) + torch.sum(y_true * y_true, [2, 3, 4]) + 0.001
    dice = torch.mean((2.0 * intersection + 0.001) / union)
    return dice


def dice_loss(y_conv, y_true):
    """Compute dice among **positive** labels to avoid unbalance.
    Argument:
        y_true: [batch_size * depth * height * width, (1)] (torch.cuda.LongTensor)
        y_conv: [batch_size * depth * height * width,  2 ] (torch.cuda.FloatTensor)
    """
    intersection = torch.sum(y_conv * y_true, [2, 3, 4])
    # `dim = 0` for Tensor result
    union = torch.sum(y_conv * y_conv, [2, 3, 4]) + torch.sum(y_true * y_true, [2, 3, 4]) + 0.001
    dice = torch.mean((2.0 * intersection + 0.001) / union)
    return 1 - torch.clamp(dice, 0.0, 1.0 - 1e-7)
