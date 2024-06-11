import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FF
from Reinforce import Policy
from torch.nn import functional as nn
from torch.nn.modules import Module
from torch.optim import Adam

class Mask():
    def __init__(self, num_masks, img_size):
        # self.num_masks = num_masks
        self.img_size = img_size

    def init_mask(self, num_dat, num_masks):
        mask_inds = torch.zeros([num_dat, num_masks])
        mask_size = int(np.ceil(self.img_size / num_masks)) + 6
        mask_all_inds = np.random.permutation(self.img_size * self.img_size)
        mask_inds[:, 0] = int(mask_all_inds[0])
        mask_inds[:, 1] = int(mask_all_inds[5])
        mask_inds[:, 2] = int(mask_all_inds[7])

        return mask_inds, mask_size

    def built_mask(self, mask_inds, mask_size):
        mask = torch.zeros([mask_inds.shape[0], 3, self.img_size, self.img_size])
        for i in range(mask_inds.shape[0]):
            mask[i, :, int(mask_inds[i, 0, 0]):int(mask_inds[i, 0, 0] + mask_size),int(mask_inds[i, 0, 1]): int(mask_inds[i, 0, 1] + mask_size)] = 1
            mask[i, :, int(mask_inds[i, 1, 0]):int(mask_inds[i, 1, 0] + mask_size),
            int(mask_inds[i, 1, 1]): int(mask_inds[i, 1, 1] + mask_size)] = 1
            mask[i, :, int(mask_inds[i, 2, 0]):int(mask_inds[i, 2, 0] + mask_size),
            int(mask_inds[i, 2, 1]): int(mask_inds[i, 2, 1] + mask_size)] = 1
        return mask

    def img_masks(self, mask_inds, img, mask_size, num_masks):
        Masks = []
        for i in range(num_masks):
            maskk = torch.zeros([mask_inds.shape[0], 3, mask_size, mask_size])
            for j in range(mask_inds.shape[0]):
                pp = img[j, :, int(mask_inds[j, 0, 0]):int(mask_inds[j, 0, 0] + mask_size),
                     int(mask_inds[j, 0, 1]): int(mask_inds[j, 0, 1] + mask_size)]
                maskk[j, :, :] = FF.resize(pp, [mask_size, mask_size])
            Masks.append(maskk)
        return Masks

    def reshape_mask(self, mask):
        new_mask = np.zeros([mask.shape[0], mask.shape[2], mask.shape[3], mask.shape[1]])
        # new_mask= np.zeros([mask.shape[1], mask.shape[2], mask.shape[0]])
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                new_mask[i, :, :, j] = mask[i, j, :, :]
        return new_mask


def one_to_two(num, img_size, num_masks):
    num = num.to('cpu')
    inds = torch.zeros([num.shape[0],num_masks, 2], dtype=int)
    for i in range(num.shape[0]):
        for j in range(num_masks):
           if num[i,j] % img_size == 0:
               inds[i,j ,0] = torch.floor_divide(num[i,j] , img_size)
               inds[i,j, 1] = 0
           else:
               inds[i,j, 0] = torch.floor_divide(num[i,j] , img_size)
               inds[i,j, 1] = num[i,j] % img_size


    return inds