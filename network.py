import sys

# third party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dim=3):
        """
        + Instantiate modules: conv-relu-norm
        + Assign them as member variables
        """
        super(conv_block, self).__init__()
        conv_type = getattr(nn, 'Conv%dd'%dim)
        self.conv = conv_type(in_channels, out_channels, kernel_size=3,stride=stride, padding=1)
        self.leakyrelu = nn.ReLU()
        # with learnable parameters
        # self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        return self.leakyrelu(self.conv(x))


class unet_core(nn.Module):
    def __init__(self, ndims, enc_nf, dec_nf, full_size=True, src_feats=1):
      super(unet_core, self).__init__()
      assert ndims in [2, 3], "ndims should be one of 2, or 3. found: %d" % ndims
      # upsample_layer = getattr(nn, 'UpSample')
      self.full_size = full_size
      self.conv_1 = conv_block(src_feats*2, enc_nf[0], stride=2, dim = ndims)
      self.conv_2 = conv_block(enc_nf[0], enc_nf[1], stride=2, dim = ndims)
      self.conv_3 = conv_block(enc_nf[1], enc_nf[2], stride=2, dim = ndims)
      self.conv_4 = conv_block(enc_nf[2], enc_nf[3], stride=2, dim = ndims)
      self.deconv_4 = conv_block(enc_nf[3], dec_nf[0], stride=1, dim = ndims)
      self.deconv_3 = conv_block(dec_nf[0]+enc_nf[2],dec_nf[1], stride=1, dim = ndims)
      self.deconv_2 = conv_block(dec_nf[1]+enc_nf[1],dec_nf[2], stride=1, dim = ndims)
      self.deconv_1 = conv_block(dec_nf[2]+enc_nf[0],dec_nf[3], stride=1, dim = ndims)
      #self.conv_5 = conv_block(dec_nf[3],dec_nf[4], stride=1, dim = ndims)
      self.conv_6 = None
      self.conv_7 = None
      if self.full_size:
          #self.conv_6 = conv_block(dec_nf[4]+src_feats*2,dec_nf[5], stride=1, dim = ndims)
          self.conv_6 = conv_block(dec_nf[4],dec_nf[5], stride=1, dim = ndims)
      # optional convolution at output resolution (used in voxelmorph-2)
      if len(dec_nf) == 7:
          self.conv_7 = conv_block(dec_nf[5],dec_nf[6], stride=1, dim = ndims)
      if ndims==2:
          self.mode = 'bilinear'
      elif ndims==3:
          self.mode = 'trilinear'
      else: self.mode = 'nearest'

    def forward(self, src, tgt):
      x = torch.cat((src,tgt),1)
      conv_1 = self.conv_1(x)
      conv_2 = self.conv_2(conv_1)
      conv_3 = self.conv_3(conv_2)
      conv_4 = self.conv_4(conv_3)
      deconv = F.interpolate(self.deconv_4(conv_4), scale_factor=2, mode=self.mode, align_corners=True)
      deconv = torch.cat((deconv,conv_3),1)
      deconv = F.interpolate(self.deconv_3(deconv), scale_factor=2, mode=self.mode, align_corners=True)
      deconv = torch.cat((deconv,conv_2),1)
      deconv = F.interpolate(self.deconv_2(deconv), scale_factor=2, mode=self.mode, align_corners=True)
      deconv = torch.cat((deconv,conv_1),1)
      deconv = self.deconv_1(deconv)
      if self.conv_6 is not None:
        deconv = F.interpolate(deconv, scale_factor=2, mode=self.mode, align_corners=True)
        #deconv = torch.cat((deconv,x),1)
        deconv = self.conv_6(deconv)
      if self.conv_7 is not None:
        deconv = self.conv_7(deconv)
      return deconv
      
class conv_block_no_pad(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dim=3):
        """
        + Instantiate modules: conv-relu-norm
        + Assign them as member variables
        """
        super(conv_block_no_pad, self).__init__()
        conv_type = getattr(nn, 'Conv%dd'%dim)
        self.conv = conv_type(in_channels, out_channels, kernel_size=3,stride=stride, padding=0)
        self.leakyrelu = nn.ReLU()
        # with learnable parameters
        # self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        return self.leakyrelu(self.conv(x))

class conv_block_2(nn.Module):
    def __init__(self, in_channels, out_channels, dim=3):
        """
        + Instantiate modules: conv-relu-norm
        + Assign them as member variables
        """
        super(conv_block_2, self).__init__()
        #conv_type = getattr(nn, 'Conv%dd'%dim)
        self.conv1 = conv_block_no_pad(in_channels, in_channels,stride=1,dim=dim)
        self.conv2 = conv_block_no_pad(in_channels, out_channels,stride=2,dim=dim)
        # with learnable parameters
        # self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        #self.
    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return x
        
def crop(large, small):
    """large / small with shape [batch_size, channels, depth, height, width]"""

    l, s = large.size(), small.size()
    offset = [0, 0, (l[2] - s[2]) // 2, (l[3] - s[3]) // 2, (l[4] - s[4]) // 2]
    return large[..., offset[2]: offset[2] + s[2], offset[3]: offset[3] + s[3], offset[4]: offset[4] + s[4]]
    
class deconv_block_2(nn.Module):
    def __init__(self, in_channels, out_channels, dim=3):
        """
        + Instantiate modules: conv-relu-norm
        + Assign them as member variables
        """
        super(deconv_block_2, self).__init__()
        self.conv2 = conv_block_no_pad(in_channels, out_channels,stride=1,dim=dim)
        if dim==2:
          self.mode = 'bilinear'
        elif dim==3:
            self.mode = 'trilinear'
        else: self.mode = 'nearest'
    def forward(self, x, en):
        x = F.interpolate(x, scale_factor=2, mode=self.mode, align_corners=True)
        x = torch.cat((x,crop(en,x).contiguous()),1)
        x = self.conv2(x)
        return x

      
class unet_nopadding(nn.Module):
    def __init__(self, ndims, enc_nf, src_feats=1):
      super(unet_nopadding, self).__init__()
      assert ndims in [2, 3], "ndims should be one of 2, or 3. found: %d" % ndims
      # upsample_layer = getattr(nn, 'UpSample')
      #self.full_size = full_size
      self.conv_in = conv_block_no_pad(src_feats*2, enc_nf[0], dim = ndims)
      self.conv_1 = conv_block_no_pad(enc_nf[0], enc_nf[0], dim = ndims)
      self.conv_2 = conv_block_2(enc_nf[0], enc_nf[1], dim = ndims)
      self.conv_3 = conv_block_2(enc_nf[1], enc_nf[2], dim = ndims)
      self.conv_4 = conv_block_2(enc_nf[2], enc_nf[3], dim = ndims)
      self.deconv_4 = deconv_block_2(enc_nf[2]+enc_nf[3], enc_nf[2], dim = ndims)
      self.deconv_3 = deconv_block_2(enc_nf[1]+enc_nf[2],enc_nf[1], dim = ndims)
      self.deconv_2 = deconv_block_2(enc_nf[0]+enc_nf[1],enc_nf[0], dim = ndims)
      self.deconv_1 = conv_block_no_pad(enc_nf[0],enc_nf[0], stride=1, dim = ndims)

    def forward(self, src, tgt):
      x = torch.cat((src,tgt),1)
      x = self.conv_in(x)
      conv_1 = self.conv_1(x)
      conv_2 = self.conv_2(conv_1)
      conv_3 = self.conv_3(conv_2)
      conv_4 = self.conv_4(conv_3)
      deconv =self.deconv_4(conv_4, conv_3)
      deconv = self.deconv_3(deconv, conv_2)
      deconv = self.deconv_2(deconv, conv_1)
      deconv = self.deconv_1(deconv)
      return deconv

