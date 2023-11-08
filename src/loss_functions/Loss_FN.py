import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyL1Loss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_l1=1.0):
        super(CrossEntropyL1Loss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_l1 = weight_l1
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        l1_loss = self.l1_loss(inputs, targets)
        loss = self.weight_ce * ce_loss + self.weight_l1 * l1_loss
        return loss
    

class WeightedCrossEntropyWithL1Loss(nn.Module):
    def __init__(self, weight_decay=0.01):
        super(WeightedCrossEntropyWithL1Loss, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, inputs, targets, model):
        cross_entropy_loss = F.cross_entropy(inputs, targets)
        l1_loss = torch.tensor(0.0).to(inputs.device)

        tot = 0
        for name, param in model.named_parameters():
            tot+=len(param.flatten())
            if 'weight' in name:
                l1_loss += torch.inner(param.flatten(), param.flatten())
                # torch.sqrt(torch.inner(param.flatten(), param.flatten()))
        l1_loss=torch.sqrt(l1_loss)

        weighted_l1_loss = l1_loss / tot
        loss = cross_entropy_loss + weighted_l1_loss/100

        return loss