import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GADice(nn.Module):
    def __init__(self, GA=True):
        self.GA = GA
        super(GADice, self).__init__()

    def _one_hot_encoder(self, input_tensor):        
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, cls, score, target, weighted_pixel_map=None):
        target = target.float()
        if weighted_pixel_map is not None:
            target = target * weighted_pixel_map
        smooth = 1e-10

        intersection = 2 * torch.sum(score * target) + smooth
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        
        return loss

    def forward(self, inputs, target, argmax=False, one_hot=True, weight=None, softmax=False, weighted_pixel_map=None):
        self.n_classes = inputs.size()[1]
        if len(inputs.size()) == len(target.size())+1:
            target = target.unsqueeze(1)
        
        if softmax:
            inputs = F.softmax(inputs, dim=1)
        if argmax:
            target = torch.argmax(target, dim=1)
        if one_hot:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        smooth = 1e-10
        loss = 0.0
        for i in range(0, self.n_classes):
            if torch.sum(target[:, i]) > 0:
                dice_loss = self._dice_loss(i, inputs[:, i], target[:, i], weighted_pixel_map)
            else:
                if self.GA:
                    beta = inputs[:, i] / (torch.sum(1 - target[:, i]))
                    dice_loss = torch.sum(beta.detach() * inputs[:, i])                
            loss += dice_loss * weight[i]
        
        return loss / self.n_classes
        
        
        
        
class GACE(torch.nn.CrossEntropyLoss):
    def __init__(self, weight=None, ignore_index=-100, k=10, gama=0.5):
        self.k = k
        self.gama = gama
        super(GACE, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        target = target.long()
        self.n_classes = inp.size()[1]

        i0 = 1
        i1 = 2
        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, self.n_classes)
        target = target.view(-1,)
        res = super(GACE, self).forward(inp, target)      
  
  
        n_instance = np.prod(res.shape)
        res, indices = torch.topk(res.view((-1, )), int(n_instance * self.k / 100), sorted=False)      
        target = torch.gather(target, 0, indices)        
        assert res.size() == target.size(), 'predict & target shape do not match'
        
        bg_w = np.power(int(n_instance * self.k / 100), self.gama)
        loss = 0.0
        smooth = 1e-10
        for i in range(0, self.n_classes):        
            target_cls = (target == i).float()  
            w = torch.pow(torch.sum(target_cls) + smooth, 1-self.gama) * bg_w
            loss_cls = torch.sum(res * target_cls) / (w + smooth)
            loss += loss_cls

        return loss