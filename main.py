from torch.utils.data import DataLoader
from util import *
import warnings
from torch.optim.lr_scheduler import StepLR
warnings.filterwarnings('ignore')
CUDA_LAUNCH_BLOCKING = 1
from util import *
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from prototypical_batch_sampler import PrototypicalBatchSampler
from protonet1 import ProtoNet
from parser_util import get_parser
import argparse
import torch
import numpy as np
from mini_imagenet import MiniImageNet, MiniImageNet2
from torch.distributions import Bernoulli, Categorical
import os
import torch.nn as nn
from torch.nn import functional as F
from MASKS import Mask, one_to_two
import torch.optim as optim
from samplers import CategoriesSampler
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')
from torch.nn.modules.module import Module
CUDA_LAUNCH_BLOCKING = 1
from torchvision import transforms
import torchvision.models as models
from torchvision.transforms import transforms


class randomly_select_patch():
    def __init__(self, num_masks, img_size):
        self.num_masks = num_masks
        self.img_size = img_size

    def select_mask(self):
        mask_size = 25
        h_ms = np.ceil(mask_size / 2)
        msk_co = []
        for j in range(self.num_masks):
            ind = (one_to_two2((random.randint(0, self.img_size * self.img_size)), self.img_size))
            if ind[0] + h_ms > self.img_size - 1:
                ind[0] = self.img_size - 1 - h_ms
            elif ind[0] - h_ms < 0:
                ind[0] = h_ms

            if ind[1] + h_ms > self.img_size - 1:
                ind[1] = self.img_size - 1 - h_ms
            elif ind[1] - h_ms < 0:
                ind[1] = h_ms
            msk_co.append(ind)
        return msk_co, h_ms

    def patch(self, img, msk_co, h_ms):
        h_ms = int(h_ms)

        ############################# check condition for mask ###########################################
        for i in range(self.num_masks):
            if msk_co[i][0] + h_ms > self.img_size - 1:
                msk_co[i][0] = self.img_size - 1 - h_ms
            elif msk_co[i][0] - h_ms < 0:
                msk_co[i][0] = h_ms

            if msk_co[i][1] + h_ms > self.img_size - 1:
                msk_co[i][1] = self.img_size - 1 - h_ms
            elif msk_co[i][1] - h_ms < 0:
                msk_co[i][1] = h_ms
        ###################################################################################################
        patch1 = img[:, :, int(msk_co[0][0]) - h_ms:int(msk_co[0][0]) + h_ms,
                 int(msk_co[1][1]) - h_ms:int(msk_co[1][1]) + h_ms]
        patch2 = img[:, :, int(msk_co[1][0]) - h_ms:int(msk_co[1][0]) + h_ms,
                 int(msk_co[1][1]) - h_ms:int(msk_co[1][1]) + h_ms]
        patch3 = img[:, :, int(msk_co[2][0]) - h_ms:int(msk_co[2][0]) + h_ms,
                 int(msk_co[2][1]) - h_ms:int(msk_co[2][1]) + h_ms]
        return patch1, patch2, patch3

    def masked_img(self, img, msk_co, h_ms):
        h_ms = int(h_ms)
        msk_co = msk_co[0]
        ############################# check condition for mask ###########################################
        for i in range(self.num_masks):
            if msk_co[0] + h_ms > self.img_size - 1:
                msk_co[0] = self.img_size - 1 - h_ms
            elif msk_co[0] - h_ms < 0:
                msk_co[0] = h_ms

            if msk_co[1] + h_ms > self.img_size - 1:
                msk_co[1] = self.img_size - 1 - h_ms
            elif msk_co[1] - h_ms < 0:
                msk_co[1] = h_ms
        masked_img = torch.zeros_like(img)
        ###################################################################################################
        masked_img[:, int(msk_co[0]) - h_ms:int(msk_co[0]) + h_ms,
        int(msk_co[1]) - h_ms:int(msk_co[1]) + h_ms] = img[:, int(msk_co[0]) - h_ms:int(msk_co[0]) + h_ms,
                                                       int(msk_co[1]) - h_ms:int(msk_co[1]) + h_ms]
        # masked_img[:, int(msk_co[0]) - h_ms:int(msk_co[0]) + h_ms,
        # int(msk_co[1]) - h_ms:int(msk_co[1]) + h_ms] = img[:, int(msk_co[0]) - h_ms:int(msk_co[0]) + h_ms,
        #                                                      int(msk_co[1]) - h_ms:int(msk_co[1]) + h_ms]
        # masked_img[:, int(msk_co[0]) - h_ms:int(msk_co[0]) + h_ms,
        # int(msk_co[1]) - h_ms:int(msk_co[1]) + h_ms] = img[:, int(msk_co[0]) - h_ms:int(msk_co[2][0]) + h_ms,
        #                                                      int(msk_co[1]) - h_ms:int(msk_co[2][1]) + h_ms]
        return masked_img

    def masked_img2(self, img, msk_co, mask_size):
        ############ return the patch for one image #############
        # h_ms = int(mask_size)
        masked_img = torch.zeros_like(img)
        masked_img[:, int(msk_co[0, 0]):int(msk_co[0, 0] + int(mask_size[0])),
        int(msk_co[0, 1]):int(msk_co[0, 1] + int(mask_size[0]))] = img[:, int(msk_co[0, 0]):int(
            msk_co[0, 0] + int(mask_size[0])), int(msk_co[0, 1]):int(msk_co[0, 1] + int(mask_size[0]))].clone()
        masked_img[:, int(msk_co[1, 0]):int(msk_co[1, 0] + int(mask_size[1])),
        int(msk_co[1, 1]):int(msk_co[1, 1] + int(mask_size[1]))] = img[:, int(msk_co[1, 0]):int(
            msk_co[1, 0] + int(mask_size[1])), int(msk_co[1, 1]):int(msk_co[1, 1] + int(mask_size[1]))].clone()
        masked_img[:, int(msk_co[2, 0]):int(msk_co[2, 0] + int(mask_size[2])),
        int(msk_co[2, 1]):int(msk_co[2, 1] + int(mask_size[2]))] = img[:, int(msk_co[2, 0]):int(
            msk_co[2, 0] + int(mask_size[2])), int(msk_co[2, 1]):int(msk_co[2, 1] + int(mask_size[2]))].clone()

        # cropped_img = img[:, int(msk_co[0]):int(msk_co[0]) + h_ms, int(msk_co[1]):int(msk_co[1]) + h_ms]
        # print(cropped_img.shape)
        # show(cropped_img, 1)
        # T = transforms.Compose([transforms.Resize(size=(84, 84))])
        # sh = int(np.ceil(84 / cropped_img.shape[0]))
        # m = cropped_img.repeat(1, sh, sh)
        # ppp = m[:, 0:84, 0:84]
        # ppp = fn.resize(cropped_img, size=[84, 84])
        # ppp = T(cropped_img)

        # print('pppppppppppppp', ppp)
        return masked_img

    def conca(self, img, msk_co, h_ms):
        size = img.shape
        h_ms = int(h_ms)
        msk_co = msk_co[0]
        ############################# check condition for mask ###########################################
        for i in range(self.num_masks):
            if msk_co[0] + h_ms > self.img_size - 1:
                msk_co[0] = self.img_size - 1 - h_ms
            elif msk_co[0] - h_ms < 0:
                msk_co[0] = h_ms

            if msk_co[1] + h_ms > self.img_size - 1:
                msk_co[1] = self.img_size - 1 - h_ms
            elif msk_co[1] - h_ms < 0:
                msk_co[1] = h_ms
        masked_img = torch.zeros_like(img)

        ###################################################################################################
        m = img[:, int(msk_co[0]) - h_ms:int(msk_co[0]) + h_ms, int(msk_co[1]) - h_ms:int(msk_co[1]) + h_ms].repeat(1,
                                                                                                                    7,
                                                                                                                    7)
        im_s = m[:, 0:84, 0:84]
        return im_s


RSP = randomly_select_patch(1, 84)


def sampls(input, target, n_support):
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda:2 too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    support_samples = torch.stack([input_cpu[idx_list] for idx_list in support_idxs])
    S_s = torch.zeros(
        [support_samples.shape[0] * support_samples.shape[1], support_samples.shape[2], support_samples.shape[3],
         support_samples.shape[4]])
    L_s = []
    ll = 0
    for i in range(support_samples.shape[0]):
        for j in range(support_samples.shape[1]):
            S_s[ll, :, :, :] = support_samples[i][j]
            ll += 1
            L_s.append(i)

    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]

    return query_samples, S_s, L_s


def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


def show(img, label):
    img = np.array(img.to('cpu'))
    new_data = np.zeros([img.shape[1], img.shape[2], img.shape[0]])
    for j in range(img.shape[0]):
        if j == 0:
            mean = 0.485
            std = 0.229
        elif j == 1:
            mean = 0.456
            std = 0.224
        else:
            mean = 0.406
            std = 0.225
        new_data[:, :, j] = (img[j, :, :] * std) + mean
    plt.imshow(new_data)
    plt.title(str(label))
    plt.show()
    return new_data


def show2(img1, img2, label):
    img1 = np.array(img1.to('cpu'))
    img2 = np.array(img2.to('cpu'))
    new_data1 = np.zeros([img1.shape[1], img1.shape[2], img1.shape[0]])
    new_data2 = np.zeros([img2.shape[1], img2.shape[2], img2.shape[0]])
    for j in range(img1.shape[0]):
        if j == 0:
            mean = 0.485
            std = 0.229
        elif j == 1:
            mean = 0.456
            std = 0.224
        else:
            mean = 0.406
            std = 0.225
        new_data1[:, :, j] = (img1[j, :, :] * std) + mean

    for j in range(img2.shape[0]):
        if j == 0:
            mean = 0.485
            std = 0.229
        elif j == 1:
            mean = 0.456
            std = 0.224
        else:
            mean = 0.406
            std = 0.225
        new_data2[:, :, j] = (img2[j, :, :] * std) + mean

    fig, axes = plt.subplots(nrows=1, ncols=2)

    # Show the first image in the first subplot
    axes[0].imshow(new_data1)
    axes[0].set_title('Image 1')

    # Show the second image in the second subplot
    axes[1].imshow(new_data2)
    axes[1].set_title('Image 2')

    # plt.imshow(new_data)
    # plt.title(str(label))
    plt.show()


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, mode):
    dataset = OmniglotDataset(mode=mode, root=opt.dataset_root)
    n_classes = len(np.unique(dataset.y))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise (Exception('There are not enough classes in the dataset in order ' +
                         'to satisfy the chosen classes_per_it. Decrease the ' +
                         'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:2' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ProtoNet().to(device)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


class Env(Mask):
    def __init__(self, num_masks, img_size):
        super().__init__(num_masks, img_size)
        self.img_size = img_size
        self.num_masks = num_masks
        self.mask_size = 28

    def reset(self, imgs, RSP):
        ms = self.mask_size
        masked_img = torch.zeros_like(imgs)

        msk_co = torch.zeros(imgs.shape[0], 3, 2)
        msk_co[:, 0, 0] = 0
        msk_co[:, 0, 1] = 0

        msk_co[:, 1, 0] = 0
        msk_co[:, 1, 1] = 28

        msk_co[:, 2, 0] = 0
        msk_co[:, 2, 1] = 55
        masked_img[:, 0:28, 0:28] = imgs[:, 0:28, 0:28].clone()
        masked_img[:, 28:56, 0:28] = imgs[:, 28:56, 0:28].clone()
        masked_img[:, 56:84, 0:28] = imgs[:, 56:84, 0:28].clone()
        # ms = self.mask_size 
        # masked_img = torch.zeros_like(imgs)

        # msk_co = torch.zeros((imgs.shape[0],3, 2), dtype = torch.int)
        # msk_co[:,0,0] = int(random.sample(range(28, 55),1)[0])

        # msk_co[:,0,1] = int(random.sample(range(28, 55),1)[0])

        # msk_co[:,1,0] = int(random.sample(range(28, 55),1)[0])
        # msk_co[:,1,1] = int(random.sample(range(28, 55),1)[0])

        # msk_co[:,2,0] = int(random.sample(range(28, 55),1)[0])
        # msk_co[:,2,1] = int(random.sample(range(28, 55),1)[0])
        # masked_img[:, msk_co[0,0,0]:msk_co[0,0,0]+28,msk_co[0,0,1]:msk_co[0,0,1]+28] = imgs[0, msk_co[0,0,0]:msk_co[0,0,0]+28,msk_co[0,0,1]:msk_co[0,0,1]+28].clone()
        # masked_img[:, msk_co[0,1,0]:msk_co[0,1,0]+28,msk_co[0,1,1]:msk_co[0,1,1]+28] = imgs[0, msk_co[0,1,0]:msk_co[0,1,0]+28,msk_co[0,1,1]:msk_co[0,1,1]+28].clone()
        # masked_img[:, msk_co[0,2,0]:msk_co[0,2,0]+28,msk_co[0,2,1]:msk_co[0,2,1]+28] = imgs[0, msk_co[0,2,0]:msk_co[0,2,0]+28,msk_co[0,2,1]:msk_co[0,2,1]+28].clone()
        # for i in range(imgs.shape[0]):
        # m, h_ms = RSP.select_mask()
        # msk_co[i, :] = m[0]
        # masked_img[i] = RSP.masked_img2(imgs[i], msk_co[i, :], h_ms)
        h = np.ceil(14 * 2)
        return masked_img, h.repeat(imgs.shape[0] * 3).reshape([imgs.shape[0], 3]), msk_co

    def step(self, act, imgs, msk_co, mask_size, RSP, labels, model2, train_policy, episode):
        num_dat = imgs.shape[0]
        new_imgs = imgs.clone()
        mask_size = torch.tensor(mask_size, dtype=torch.int)

        step = 1
        for i in range(num_dat):
            for j in range(3):

                if act[i, j] == 0:
                    ########## go up ###########
                    if msk_co[i, j, 1] - step >= 0:
                        msk_co[i, j, 1] = msk_co[i, j, 1] - step
                elif act[i, j] == 1:
                    ########## No change ###########
                    msk_co[i, j] = msk_co[i, j]

                elif act[i, j] == 2:
                    ########## Go down ###########
                    if msk_co[i, j, 1] + step + mask_size[i, j] < imgs.shape[1]:
                        msk_co[i, j, 1] = msk_co[i, j, 1] + step

                elif act[i, j] == 3:
                    ########## go left ###########
                    if msk_co[i, j, 0] - step >= 0:
                        msk_co[i, j, 0] = msk_co[i, j, 0] - step

                elif act[i, j] == 4:
                    ########## Go right ###########
                    if msk_co[i, j, 0] + step + mask_size[i, j] < imgs.shape[2]:
                        msk_co[i, j, 0] = msk_co[i, j, 0] + step

                    if torch.floor(mask_size[i, j] * 1.5) < 35 and msk_co[i, j, 0] + 1 + torch.floor(
                            mask_size[i, j] * 1.5) < \
                            imgs.shape[2]:
                        mask_size[i, j] = torch.floor(mask_size[i, j] * 1.5)
            new_imgs[i] = RSP.masked_img2(imgs[i], msk_co[i], mask_size[i])
            return new_imgs, mask_size

    # def step(self, act, imgs, RSP, h_ms, labels, model,support_samples,query_samples, train_policy):
    #     num_dat = imgs.shape[0]
    #     # print(act.shape)
    #     # msk_co = one_to_two(act, self.img_size, self.num_masks).to('cuda:2')
    #     # print(act.shape)
    #     msk_co = act.to('cuda')
    #     masked_img = torch.zeros_like(imgs)
    #     for i in range(num_dat):
    #         masked_img[i] = RSP.conca(imgs[i], msk_co[i, :, :], h_ms)
    #     if train_policy:
    #         reward, ACC = Reward(model, support_samples,query_samples,masked_img,labels)
    #         done = 1
    #         return masked_img, reward, done, ACC
    #     else:
    #         done = 1
    #         return masked_img, done


def Reward(model, imgs, samples, msk_co, mask_size, L, buffer_size, buffer_inf, done, data_inds, episode):
    model.eval()
    y = L
    y_pred = model(samples)
    output = F.softmax(y_pred, dim=1)
    _, y_p = y_pred.max(1)
    loss = criterion(y_pred, y)
    acc = (y_p.eq(y).float().mean() * 100).item()
    reward = acc
    ccccccccccc = 0
    # print(output.max(1)[0])
    # ?print(acc)
    if done == 1:
        for i in range(samples.shape[0]):
            if output[i, y[i]] >= 0.85:
                ccccccccccc += 1
                # show2(imgs[i] ,samples[i], y)
                p1 = imgs[i, :, int(msk_co[i, 0, 0]):int(msk_co[i, 0, 0] + mask_size[i, 0]),
                     int(msk_co[i, 0, 1]):int(msk_co[i, 0, 1] + mask_size[i, 0])]
                p2 = imgs[i, :, int(msk_co[i, 1, 0]):int(msk_co[i, 1, 0] + mask_size[i, 1]),
                     int(msk_co[i, 1, 1]):int(msk_co[i, 1, 1] + mask_size[i, 1])]
                p3 = imgs[i, :, int(msk_co[i, 2, 0]):int(msk_co[i, 2, 0] + mask_size[i, 2]),
                     int(msk_co[i, 2, 1]):int(msk_co[i, 2, 1] + mask_size[i, 2])]
                save_buff = torch.cat([p1, p2, p3], dim=1)
                buffer_inf = fill_buffer(buffer_inf, save_buff, int(y[i]), data_inds[i], output[i, y[i]], episode)
        # print('ccccccccccc',ccccccccccc)
    return reward, acc, buffer_inf


def fill_buffer(buffer_inf, img, img_label, data_inds, output, episode):
    num = 0.5
    if episode >= 240:
        num = 0.4
    if episode >= 450:
        num = 0.3

    if random.random() >= num:
        if buffer_inf['confidence'][data_inds - 1] < output:
            buffer_inf['buffer_mask'][data_inds - 1] = 1
            buffer_inf['labels'][data_inds - 1] = img_label
            buffer_inf['confidence'][data_inds - 1] = output

            torch.save(img, 'processed_data4/' + str(data_inds) + '.pt')
    else:
        buffer_inf['buffer_mask'][data_inds - 1] = 1
        buffer_inf['labels'][data_inds - 1] = img_label
        buffer_inf['confidence'][data_inds - 1] = output

        torch.save(img, 'processed_data4/' + str(data_inds) + '.pt')

    return buffer_inf


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''

    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)  # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


# class Reinforce(nn.Module):
#   def __init__(self, x_dim, hid_dim, z_dim, num_actions=7, use_bias=True):
#      ###### actions: up-down-right-nochange---- left-1/2-*2-1
#     super(Reinforce, self).__init__()
#
#       self.num_actions = num_actions
#
#       super(Reinforce, self).__init__()
#      self.encoder = nn.Sequential(
#         conv_block(x_dim, hid_dim),
#        conv_block(hid_dim, hid_dim),
#       conv_block(hid_dim, hid_dim),
#      conv_block(hid_dim, z_dim),
# )

# self.conv = conv_block(3, 6)
# self.image_linear = nn.Linear(10584, 560)
# self.relu = nn.LeakyReLU()
# self.linearx = nn.Linear(1600, 560)
# self.actor_linear1 = nn.Linear(560 * 2, 128)
# self.actor_linear2 = nn.Linear(128, 7)

# def forward(self, state, state2):
#  x = Variable(state)
# x2 = Variable(state2)
# x2 = self.conv(state2)
# x2 = self.image_linear(x2.view(x2.size(0), -1))
# print('x2.shape',x2.shape) #### 50*560
# x = self.encoder(x)
# x = self.relu(self.linearx(x.view(x.shape[0], -1)))
# print('x.shape',x.shape) ####
# x = torch.cat((x2, x), dim=1)
# policy_dist = self.relu(self.actor_linear1(x))

# policy_dist = self.actor_linear2(policy_dist)

# print('policy_dist.shape', policy_dist.shape)
# policy_dist = F.softmax(policy_dist, dim=1)

# return policy_dist

# return x


class Reinforce(nn.Module):
    def __init__(self, x_dim, hid_dim, z_dim, num_actions=5, use_bias=True):
        ###### actions: up-down-right-nochange---- left-1/2-*2-1
        super(Reinforce, self).__init__()

        self.num_actions = num_actions
        self.layer = models.resnet50(pretrained=True)
        fc2 = nn.Linear(2048, 560)
        self.layer.fc = fc2
        self.conv = conv_block(3, 6)
        self.image_linear = nn.Linear(10584, 560)
        self.relu = nn.LeakyReLU()
        self.actor_linear1 = nn.Linear(560 * 2, 128)
        self.actor_linear2 = nn.Linear(128, 5 * 3)

    def forward(self, state, state2):
        x = Variable(state)
        x2 = Variable(state2)
        x2 = self.conv(x2)
        x2 = self.image_linear(x2.view(x2.size(0), -1))
        x = self.layer(x)
        x = self.relu(x)
        x = torch.cat((x2, x), dim=1)
        policy_dist = self.relu(self.actor_linear1(x))
        policy_dist = self.actor_linear2(policy_dist).reshape([policy_dist.shape[0], 3, 5])
        policy_dist = F.softmax(policy_dist, dim=2)

        return policy_dist


def one_to_two2(num, img_size):
    inds = torch.zeros([2], dtype=int)
    if num % img_size == 0:
        inds[0] = torch.div(num, img_size, rounding_mode='floor')
        inds[1] = 0
    else:
        inds[0] = torch.div(num, img_size, rounding_mode='floor')
        inds[1] = num % img_size

    return inds


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def train_proto2(buffer, optimiz):
    folder_path = 'processed_data4/'
    file_names = os.listdir(folder_path)
    pt_files = [f for f in file_names if f.endswith('.pt')]
    data = []
    labels = []
    for file_name in pt_files:
        # load the file

        file_path = os.path.join(folder_path, file_name)
        data.append(torch.load(file_path).to('cpu'))
        lab = int(file_name.replace('.pt', ''))
        labels.append(buffer['labels'][lab - 1])

    num_epochs = 50
    # indices = torch.randperm(len(buffer['samples'])).to(dtype=torch.long)

    # training_set = CustomTextDataset(data, labels)
    train_loader = torch.utils.data.DataLoader(list(zip(data, labels)), batch_size=20, shuffle=True)

    for i in tqdm(range(num_epochs)):

        # for j in range(np.floor(data.shape[0]/b_size)):
        #     x = data[j*b_size:(j*b_size)+b_size]
        #     y = labels[j*b_size:(j*b_size)+b_size]
        count = 0
        accc = 0
        LOSS = 0
        for (x, y) in (train_loader):
            count += 1
            x = x.to(device)
            y = y.to(device).long()
            y_pred = model2(x)
            _, y_p = y_pred.max(1)
            loss = criterion(y_pred, y)
            loss.backward()
            optimiz.step()
            acc = (y_p.eq(y).float().mean() * 100).item()
            LOSS += loss.item()
            accc += acc

    return accc / count


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = ('cuda:2')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # print(labels.shape)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        # print('logits', exp_logits)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


sup_loss = SupConLoss(temperature=0.07, contrast_mode='all',
                      base_temperature=0.07)


def calc_clloss(p1, p2, p3, L, cl_model, temperature=0.1):
    angle = 45
    transform = transforms.Compose([
        transforms.RandomRotation(angle)])
    rotated_p1 = transform(p1)
    rotated_p2 = transform(p2)
    rotated_p3 = transform(p3)

    T1 = torch.cat([p1, p2, p3], dim=2).to('cuda:2')
    T2 = torch.cat([p1, p3, p2], dim=2).to('cuda:2')
    T3 = torch.cat([p2, p1, p3], dim=2).to('cuda:2')
    T4 = torch.cat([p2, p3, p1], dim=2).to('cuda:2')
    T5 = torch.cat([p3, p2, p1], dim=2).to('cuda:2')
    T6 = torch.cat([p3, p1, p2], dim=2).to('cuda:2')

    TR1 = torch.cat([rotated_p1, rotated_p2, rotated_p3], dim=2).to('cuda:2')
    TR2 = torch.cat([rotated_p1, rotated_p3, rotated_p2], dim=2).to('cuda:2')
    TR3 = torch.cat([rotated_p2, rotated_p1, rotated_p3], dim=2).to('cuda:2')
    TR4 = torch.cat([rotated_p2, rotated_p3, rotated_p1], dim=2).to('cuda:2')
    TR5 = torch.cat([rotated_p3, rotated_p2, rotated_p1], dim=2).to('cuda:2')
    TR6 = torch.cat([rotated_p3, rotated_p1, rotated_p2], dim=2).to('cuda:2')

    features = torch.stack(
        [cl_model(T1), cl_model(T2), cl_model(T3), cl_model(T4), cl_model(T5), cl_model(T6), cl_model(TR1),
         cl_model(TR2), cl_model(TR3), cl_model(TR4), cl_model(TR5), cl_model(TR6)], dim=1).to('cuda:2')

    # labels =  torch.stack([L, L, L, L, L, L, L, L, L, L, L, L], dim = 1).to('cuda:2')
    # print(labels.shape)
    features = F.normalize(features, dim=1)
    labels = L
    loss = sup_loss(features, labels)

    return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--warmup', default=100, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=20, type=int,
                        help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int,
                        help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')

    device = 'cuda:2'

    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      5, 5 + 5)
    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      5, 5 + 5)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,

                              num_workers=8, pin_memory=True)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    5, 5 + 5)

    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)
    opt = get_parser().parse_args()
    testset = MiniImageNet('test')

    sampler = CategoriesSampler(testset.label,
                                2000, 5, 5 + 5)
    test_loader = DataLoader(testset, batch_sampler=sampler,
                             num_workers=8, pin_memory=True)

    opt = get_parser().parse_args()
    model = ProtoNet().to(device)
    START_LR = 0.00001

    criterion = nn.CrossEntropyLoss()

    cl_model = models.resnet50(pretrained=True).to('cuda:2')
    fc_cl = nn.Linear(2048, 256).to('cuda:2')
    cl_model.fc = fc_cl.to(device)

    optimiz_cl = torch.optim.Adam(params=cl_model.parameters(),
                                  lr=0.00001)

    optimiz = torch.optim.Adam(params=model.parameters(),
                               lr=0.00003)
    train_loss = []
    train_acc = []
    policy = Reinforce(x_dim=3, hid_dim=64, z_dim=64, num_actions=5, use_bias=True).to('cuda:2')
    policy_optimizer = optim.Adam(policy.parameters(), lr=0.000005, weight_decay=5e-4)
    scheduler = StepLR(policy_optimizer,
                       step_size=40,  # Period of learning rate decay
                       gamma=0.8)
    num_masks = 3
    img_size = 84

    env = Env(num_masks, img_size)
    train_policy = True
    train_supcon = False

    for episode in tqdm(range(2000)):

        losses = []
        rewardd = []

        if train_policy == True:
            policy.train()

            # print('C1', C1)
            tr_iter = iter(train_loader)
            vl_iter = iter(val_loader)

            ACCC = 0
            ac_loss_print = 0
            rew_print = 0
            num_batch = 0
            countt = 0
            acc22 = 0

            for batch in (tr_iter):

                countt += 1
                num_batch += 1
                x, y = batch
                x, y = x.to(device), y.to(device)

                query_samples, support_samples, L = sampls(x, y, opt.num_support_tr)
                all_samps = torch.cat([support_samples, query_samples], dim=0).to(device)

                done = False
                obs, mask_size, msk_co = env.reset(all_samps, RSP)
                rewards = []
                values = []
                log_probs = []
                entropy_term = 0
                saved_log_probs = []
                for steps in range(25):
                    policy_dist = policy.forward(obs, all_samps)
                    dist = policy_dist.to('cpu').clone()
                    m = Categorical(dist)
                    action = m.sample()
                    saved_log_probs.append(m.log_prob(action))

                    entropy = -torch.sum(torch.mean(dist.detach()) * torch.log(dist))
                    done = 0
                    new_state, mask_size = env.step(action, all_samps, msk_co, mask_size,
                                                    RSP, L, model,
                                                    train_policy, episode)

                    log_probs.append(m.log_prob(action).mean())
                    entropy_term += entropy
                    obs = new_state

                with torch.no_grad():
                    random_ids = np.random.randint(len(val_loader))
                    batche_val = next(vl_iter)
                    x_v, y_v = batche_val
                    x_v, y_v = x_v.to(device), y_v.to(device)
                    query_samples_v, support_samples_v, L_v = sampls(x_v, y_v, opt.num_support_tr)
                    all_samps_v = torch.cat([support_samples_v, query_samples_v], dim=0).to(device)

                    done = False
                    obs_v, mask_size_v, msk_co_v = env.reset(all_samps_v, RSP)
                    for steps in range(25):
                        policy_dist_v = policy.forward(obs_v, all_samps_v)
                        dist_v = policy_dist_v.to('cpu').clone()
                        m_v = Categorical(dist_v)
                        action_v = m_v.sample()
                        new_state, mask_size = env.step(action_v, all_samps_v, msk_co_v, mask_size_v,
                                                        RSP, L, model,
                                                        train_policy, episode)
                        obs = new_state

                    model.eval()

                    all_new_v = []
                    for i in range(all_samps.shape[0]):
                        p1_v = all_samps_v[i, :, int(msk_co_v[i, 0, 0]):int(msk_co_v[i, 0, 0] + mask_size_v[i, 0]),
                               int(msk_co_v[i, 0, 1]):int(msk_co_v[i, 0, 1] + mask_size_v[i, 0])]
                        p2_v = all_samps_v[i, :, int(msk_co_v[i, 1, 0]):int(msk_co_v[i, 1, 0] + mask_size_v[i, 1]),
                               int(msk_co_v[i, 1, 1]):int(msk_co_v[i, 1, 1] + mask_size_v[i, 1])]
                        p3_v = all_samps_v[i, :, int(msk_co_v[i, 2, 0]):int(msk_co_v[i, 2, 0] + mask_size_v[i, 2]),
                               int(msk_co_v[i, 2, 1]):int(msk_co_v[i, 2, 1] + mask_size_v[i, 2])]
                        save_buff_v = torch.cat([p1_v, p2_v, p3_v], dim=1)
                        all_new_v.append(save_buff_v)

                    emb_v = model(torch.stack(all_new_v))
                    support_new_v = emb_v[0:support_samples_v.shape[0]]
                    query_new_v = emb_v[support_samples_v.shape[0]:]
                    n_classes = 5
                    n_query = 5
                    support_idxs = [torch.tensor([0, 1, 2, 3, 4], dtype=torch.int).long(),
                                    torch.tensor([5, 6, 7, 8, 9], dtype=torch.int).long(),
                                    torch.tensor([10, 11, 12, 13, 14], dtype=torch.int).long(),
                                    torch.tensor([15, 16, 17, 18, 19], dtype=torch.int).long(),
                                    torch.tensor([20, 21, 22, 23, 24], dtype=torch.int).long()]
                    prototypes_v = torch.stack([support_new_v[idx_list].mean(0) for idx_list in support_idxs]).to('cpu')
                    query_samples_v = query_new_v.to('cpu')
                    dists_v = euclidean_dist(query_samples_v, prototypes_v)
                    log_p_y_v = F.log_softmax(-dists_v, dim=1).view(5, 5, -1)

                    target_inds = torch.arange(0, n_classes)
                    target_inds = target_inds.view(n_classes, 1, 1)
                    target_inds = target_inds.expand(n_classes, n_query, 1).long()
                    loss_val_v = -log_p_y_v.gather(2, target_inds).squeeze().view(-1).mean()
                    _, y_p_v = log_p_y_v.max(2)
                    acc_v = (y_p_v.eq(target_inds.view(5, 5)).float().mean() * 100).item()
                model.train()
                all_new = []
                P1 = []
                P2 = []
                P3 = []
                for i in range(all_samps.shape[0]):
                    p1 = all_samps[i, :, int(msk_co[i, 0, 0]):int(msk_co[i, 0, 0] + mask_size[i, 0]),
                         int(msk_co[i, 0, 1]):int(msk_co[i, 0, 1] + mask_size[i, 0])]
                    p2 = all_samps[i, :, int(msk_co[i, 1, 0]):int(msk_co[i, 1, 0] + mask_size[i, 1]),
                         int(msk_co[i, 1, 1]):int(msk_co[i, 1, 1] + mask_size[i, 1])]
                    p3 = all_samps[i, :, int(msk_co[i, 2, 0]):int(msk_co[i, 2, 0] + mask_size[i, 2]),
                         int(msk_co[i, 2, 1]):int(msk_co[i, 2, 1] + mask_size[i, 2])]
                    save_buff = torch.cat([p1, p2, p3], dim=1)
                    all_new.append(save_buff)
                    P1.append(p1)
                    P2.append(p2)
                    P3.append(p3)

                cl_loss = calc_clloss(torch.stack(P1), torch.stack(P2), torch.stack(P3),
                                      torch.cat((torch.tensor(L), torch.tensor(L)), dim=0), cl_model, temperature=0.1)

                # cl_loss = calc_clloss(p1, p2, p3, L)

                emb = model(torch.stack(all_new))
                support_new = emb[0:support_samples.shape[0]]
                query_new = emb[support_samples.shape[0]:]

                n_classes = 5
                n_query = 5
                support_idxs = [torch.tensor([0, 1, 2, 3, 4], dtype=torch.int).long(),
                                torch.tensor([5, 6, 7, 8, 9], dtype=torch.int).long(),
                                torch.tensor([10, 11, 12, 13, 14], dtype=torch.int).long(),
                                torch.tensor([15, 16, 17, 18, 19], dtype=torch.int).long(),
                                torch.tensor([20, 21, 22, 23, 24], dtype=torch.int).long()]
                prototypes = torch.stack([support_new[idx_list].mean(0) for idx_list in support_idxs]).to('cpu')
                query_samples = query_new.to('cpu')
                dists = euclidean_dist(query_samples, prototypes)
                log_p_y = F.log_softmax(-dists, dim=1).view(5, 5, -1)

                target_inds = torch.arange(0, n_classes)
                target_inds = target_inds.view(n_classes, 1, 1)
                target_inds = target_inds.expand(n_classes, n_query, 1).long()
                loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
                _, y_p = log_p_y.max(2)
                acc2 = (y_p.eq(target_inds.view(5, 5)).float().mean() * 100).item()

                acc22 += acc2
                rew_print += acc_v
                Qvals = acc_v
                log_probs = torch.stack(log_probs).to('cuda:2')
                policy_loss = (-log_probs * Qvals).mean()
                # policy_loss = (-log_probs ).mean()
                ac_loss = (2 * policy_loss) + loss_val + (cl_loss * 0.01)
                # ac_loss = policy_loss
                ac_loss_print += ac_loss.item()

                policy_optimizer.zero_grad()
                optimiz.zero_grad()
                optimiz_cl.zero_grad()
                ac_loss.backward()
                policy_optimizer.step()
                optimiz.step()
                optimiz_cl.step()
                # scheduler.step()







