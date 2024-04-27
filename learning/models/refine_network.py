# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
import einops.layers
import numpy as np
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(f'{code_dir}/../../../../')
from Utils import *
import torch.nn.functional as F
import torch
import torch.nn as nn
import einops
import einops.layers.torch
import cv2
from functools import partial
from network_modules import *
from Utils import *



class RefineNet(nn.Module):
  def __init__(self, cfg=None, c_in=4, n_view=1):
    super().__init__()
    self.cfg = cfg
    if self.cfg['use_BN']:
      norm_layer = nn.BatchNorm2d
      norm_layer1d = nn.BatchNorm1d
    else:
      norm_layer = None
      norm_layer1d = None

    self.encodeA = nn.Sequential(
      ConvBNReLU(C_in=c_in,C_out=64,kernel_size=7,stride=2, norm_layer=norm_layer),
      ConvBNReLU(C_in=64,C_out=128,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
    )

    self.encodeAB = nn.Sequential(
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ConvBNReLU(256,512,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
    )

    embed_dim = 512
    num_heads = 4
    self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=400)

    self.trans_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, 3),
    )

    if self.cfg['rot_rep']=='axis_angle':
      rot_out_dim = 3
    elif self.cfg['rot_rep']=='6d':
      rot_out_dim = 6
    else:
      raise RuntimeError
    self.rot_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, rot_out_dim),
    )
    self.rearrange1 = einops.layers.torch.Rearrange('b two c h w -> (b two) c h w', two = 2)
    self.rearrange2 = einops.layers.torch.Rearrange('(b two) c h w -> b (two c) h w', two = 2)


  def forward(self, A, B, amp:bool = False):
    """
    @A: (B,C,H,W)
    """
    with torch.cuda.amp.autocast(enabled=amp):
      bs = len(A)
      
      # x = torch.cat([A, B], dim=0)
      x = torch.stack([A, B], dim = 1)
      x = self.rearrange1(x)
      x = self.encodeA(x)
      ab = self.rearrange2(x)
      # a = x[:bs]
      # b = x[bs:]
      # ab = torch.cat((a, b), 1).contiguous()
      # ab = einops.rearrange(x, '(two b) c ... -> b (two c) ...', two = 2)
      # ab = x.reshape( (2,bs) + (x.shape[1:])).swapaxes(0,1).reshape(
      ab = self.encodeAB(ab)  #(B,C,H,W)

      ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0,2,1))
      trans = self.trans_head(ab).mean(dim=1)
      rot = self.rot_head(ab).mean(dim=1)
    return {'trans':trans, 'rot':rot}

def main():
  # torch._dynamo.config.verbose=True
  net = RefineNet({'use_BN': True,
                   'rot_rep': 'axis_angle'}, 6).cuda()
  net = torch.compile(net)#, fullgraph=True)
  A = torch.zeros((1,6,100,100), device='cuda',
   dtype=torch.float32)
  B = torch.zeros((1,6,100,100), device='cuda',
   dtype=torch.float32)
  net(A,B,True)

if __name__ == '__main__':
  main()