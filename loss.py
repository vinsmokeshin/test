import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np

class MSLE(nn.Module):
    """get registration accuracy(normalized cross correlation) loss between subject mask and target mask
    mean squared logarithmic error can be described as:
                MSLE = (log(p+1) - log(a+1))^2

    input subject image and target image, Batch_size*Channel*D*W*H
    return image similarity loss, 1 value
    """

    def __init__(self):
        super(MSLE, self).__init__()

    def forward(self, subject, target):
        log_subject = torch.log(subject+1)
        log_target = torch.log(target+1)
        msle = (log_subject-log_target)**2

        return torch.mean(torch.mean(torch.mean(msle, dim=[2, 3, 4]), 1))
        
        
class GradientLoss(nn.Module):
    """get registration accuracy(normalized cross correlation) loss between subject mask and target mask
    mean squared logarithmic error can be described as:
                MSLE = (log(p+1) - log(a+1))^2

    input subject image and target image, Batch_size*Channel*D*W*H
    return image similarity loss, 1 value
    """

    def __init__(self):
        super(GradientLoss, self).__init__()
        self.kernel = torch.zeros((3, 1, 3, 3, 3))
        self.kernel[0, 0, 2, 1, 1] = 0.5
        self.kernel[1, 0, 1, 2, 1] = 0.5
        self.kernel[2, 0, 1, 1, 2] = 0.5
        self.kernel[0, 0, 0, 1, 1] = -0.5
        self.kernel[1, 0, 1, 0, 1] = -0.5
        self.kernel[2, 0, 1, 1, 0] = -0.5
        self.kernel = self.kernel.cuda()

    def forward(self, y_pred):
        batch_size, channels, zdim, ydim, xdim = y_pred.shape
        y_pred = y_pred.view(batch_size*channels, 1, zdim, ydim, xdim)
        I_det = torch.conv3d(y_pred, self.kernel, padding=0, stride=(1, 1, 1))
        I_det = torch.pow(I_det, 2)
        I_det = torch.sum(I_det, dim=1)
        return torch.mean(torch.mean(I_det, dim=[1, 2, 3]))


class SACD(nn.Module):
    """get registration accuracy(normalized cross correlation) loss between subject mask and target mask
    mean squared logarithmic error can be described as:
                MSLE = (log(p+1) - log(a+1))^2

    input subject image and target image, Batch_size*Channel*D*W*H
    return image similarity loss, 1 value
    """

    def __init__(self):
        super(SACD, self).__init__()

    def forward(self, mask, distancemap):
        distance = torch.sum(mask*distancemap, dim=[2, 3, 4])
        square = torch.sum(mask, dim=[2, 3, 4])
        return torch.mean(torch.mean(distance/square, 1))


class DICELoss(nn.Module):
    """get registration accuracy(normalized cross correlation) loss between subject mask and target mask
    mean squared logarithmic error can be described as:
                MSLE = (log(p+1) - log(a+1))^2

    input subject image and target image, Batch_size*Channel*D*W*H
    return image similarity loss, 1 value
    """

    def __init__(self):
        super(SACD, self).__init__()

    def forward(self, mask, distancemap):
        distance = torch.sum(mask*distancemap, dim=[2, 3, 4])
        square = torch.sum(mask, dim=[2, 3, 4])
        return torch.mean(torch.mean(distance/square, 1))
        

def grad_direction(image):
    """get the image (or field in one direction) gradient

    :param: image: the input image tensor or 1-direction of the field tensor
    :return: the 3-direction(x, y, z) gradient of the image
    """

    grad_x = functional.pad(image, (0, 0, 0, 0, 0, 1))[1:, :, :] - functional.pad(image, (0, 0, 0, 0, 1, 0))[:-1, :, :]
    grad_y = functional.pad(image, (0, 0, 0, 1, 0, 0))[:, 1:, :] - functional.pad(image, (0, 0, 1, 0, 0, 0))[:, :-1, :]
    grad_z = functional.pad(image, (0, 1, 0, 0, 0, 0))[:, :, 1:] - functional.pad(image, (1, 0, 0, 0, 0, 0))[:, :, :-1]

    return grad_x / 2, grad_y / 2, grad_z / 2


class GCCLoss(nn.Module):
    def __init__(self, use_gpu=False):
        super(GCCLoss, self).__init__()
        self.use_gpu = use_gpu
        self.filter = torch.zeros((3, 1, 3, 3, 3))
        # Ux, Uy, Uz
        self.filter[0, 0, 0, 1, 1] = -1
        self.filter[1, 0, 1, 0, 1] = -1
        self.filter[2, 0, 1, 1, 0] = -1
        self.filter[0, 0, 2, 1, 1] = 1
        self.filter[1, 0, 1, 2, 1] = 1
        self.filter[2, 0, 1, 1, 2] = 1

        if use_gpu:
            self.filter = self.filter.cuda()
        self.mse = nn.MSELoss(reduction='mean')

    def get_angle(self, I, fix=False):
        batch_size, channels, zdim, ydim, xdim = I.shape
        I = I.view(batch_size * channels, 1, zdim, ydim, xdim)
        I_det = torch.conv3d(I, self.filter, padding=1, stride=(1, 1, 1))
        mask = (torch.abs(I_det).sum(dim=1)>0).float()
        #gx = torch.where(torch.abs(I_det).sum(dim=1)==0, torch.Tensor(1).float().cuda(), I_det[:, 0, ...])
        gx = I_det[:, 0, ...]
        gy = I_det[:, 1, ...]
        gz = I_det[:, 2, ...]
        gxy = gx*gx+gy*gy + 0.0001
        xyz = gxy+gz*gz
        gxy = gxy.sqrt()
        if fix:
            return gx, gy, gz, gxy, xyz, mask
        return gx, gy, gz, gxy

    def forward(self, mov, tgt):
        gx1, gy1, gz1, gxy1, xyz1, mask1 = self.get_angle(mov, True)
        gx2, gy2, gz2, gxy2, xyz2, mask2 = self.get_angle(tgt, True)
        # phi = self.mse(xy2, (gx1*gx2+gy1*gy2))
        # theta = self.mse(xyz2, (gz1*gz2+gxy1*gxy2))
        phi = (1 - (gx1*gx2+gy1*gy2) / (gxy1*gxy2))*mask1*mask2
        theta = 1 - (gz1*gz2+gxy1*gxy2) / (xyz1*xyz2).sqrt()
        return phi, theta

