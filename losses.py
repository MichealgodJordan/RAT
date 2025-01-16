import torch.nn.functional as F
from torch import nn
import torch
from torch.nn.modules.loss import _WeightedLoss
import numpy as np
from torch.autograd import Variable
from torchvision import models

def create3DsobelFilter():
    num_1, num_2, num_3 = np.zeros((3, 3))
    num_1 = [[1., 2., 1.],
             [2., 4., 2.],
             [1., 2., 1.]]
    num_2 = [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]]
    num_3 = [[-1., -2., -1.],
             [-2., -4., -2.],
             [-1., -2., -1.]]
    sobelFilter = np.zeros((3, 1, 3, 3, 3))

    sobelFilter[0, 0, 0, :, :] = num_1
    sobelFilter[0, 0, 1, :, :] = num_2
    sobelFilter[0, 0, 2, :, :] = num_3
    sobelFilter[1, 0, :, 0, :] = num_1
    sobelFilter[1, 0, :, 1, :] = num_2
    sobelFilter[1, 0, :, 2, :] = num_3
    sobelFilter[2, 0, :, :, 0] = num_1
    sobelFilter[2, 0, :, :, 1] = num_2
    sobelFilter[2, 0, :, :, 2] = num_3

    return Variable(torch.from_numpy(sobelFilter).type(torch.cuda.FloatTensor))


def sobelLayer(input):
    pad = nn.ConstantPad3d((1, 1, 1, 1, 1, 1), -1)
    kernel = create3DsobelFilter()
    act = nn.Tanh()
    paded = pad(input)
    fake_sobel = F.conv3d(paded, kernel, padding=0, groups=1)/4
    n, c, h, w, l = fake_sobel.size()
    fake = torch.norm(fake_sobel, 2, 1, True)/c*3
    fake_out = act(fake)*2-1
    return fake_out

class EdgeAwareLoss(_WeightedLoss):

    def __init__(self):
        super(EdgeAwareLoss, self).__init__()
        self.sobelLayer = sobelLayer
        self.baseloss = nn.L1Loss()

    def forward(self, input, target):
        sobelFake = self.sobelLayer(input)
        sobelReal = self.sobelLayer(target)
        return self.baseloss(sobelFake,sobelReal)

class CharbonnierLoss(nn.Module):
    def __init__(self,epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon2=epsilon*epsilon

    def forward(self,input, target):
        value=torch.sqrt(torch.pow(input-target,2)+self.epsilon2)
        return torch.mean(value)


class FocalRegionLoss(nn.Module): 
    def __init__(self, beta = 1., epsilon=1e-3):
        super(FocalRegionLoss, self).__init__()
        self.epsilon2 = epsilon*epsilon 
        self.beta = beta 


    def forward(self, input, target, mask): 

        loss_metric = F.l1_loss(input, target, reduce=False)
        weight = loss_metric.clone().detach() 
        # print('weight:',weight.shape)
        weight = weight.unsqueeze(0)
        b = weight.shape[0]

        for bi in range(b):
            mask_bi = mask[bi] #[H, W] 
            total_area = 0

            for cls_i in range(int(mask_bi.max()+1)): 
                region = mask_bi==cls_i 
                area_i = torch.sum(region)
                if area_i>0: 
                    avg_i = torch.mean(weight[bi, :, region]) 
                    weight[bi, :, region] = avg_i 
                    total_area+=area_i 
            # if total_area!=(mask_bi.shape[0]*mask_bi.shape[1]):
            #     raise ValueError("Total Area Error!")

        weight = weight/weight.max() 
        weight = torch.clamp(weight, min=0.0, max=1.0)

        return torch.mean(loss_metric*(weight*self.beta+1)) + PerceptualLoss(input, target) + AdversarialLoss(input, target)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # use pretrained vgg-19 net
        vgg = models.vgg19(pretrained=True).features.eval()
        # obtain the first 16 layers of the model
        self.vgg = vgg[:16]
        # fix the parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, generated, target):
        # extract feature
        gen_features = self.vgg(generated)
        target_features = self.vgg(target)
        # compute the loss
        loss = nn.functional.mse_loss(gen_features, target_features)
        return loss

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, discriminator_output, target_is_real):
        return self.loss(discriminator_output, target_is_real)
