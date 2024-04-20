
from typing import Optional
from torch import Tensor
from torch.nn import functional as F
import torch 
from torch.nn import Module
import torch.nn._reduction as _Reduction
import hyper_params

### Soft F1:
class my_criterion(torch.nn.BCELoss):
    def __init__(self, lambda_param = 0.001, clf = None, simple_model = None, weight: Optional[Tensor] = None, 
        size_average=None, reduce=None, reduction: str = 'sum', score_type = hyper_params.SCORE_TYPE, 
        gamma = hyper_params.FOCAL_GAMMA, alpha = hyper_params.FOCAL_ALPHA) -> None:
        super(my_criterion, self).__init__(weight, size_average, reduce, reduction)
        self.lambda_param = lambda_param
        self.clf = clf
        self.simple_model = simple_model
        self.score_type = score_type
        self.gamma = gamma
        self.alpha = alpha
        #self.roc_star = RocStar()

    def set_clf(self, clf = None):
        self.clf = clf

    def forward(self, input: Tensor, target: Tensor, prev_batch = None, epoch_num = 0, reduction = "sum",) -> Tensor:
        score_type = self.score_type
        if score_type == 'bce':
            reduction = "sum"
            new_input = input[:, 1].cpu()
            new_target = target.float().cpu()
            loss = F.binary_cross_entropy(new_input, new_target)
            if reduction == "mean":
                loss = loss.mean().cuda()
            elif reduction == "sum":
                loss = loss.sum().cuda()
            return loss

        # if score_type == 'roc-star' and epoch_num > 0:
        #     new_input = input[:, 1]#.cpu()
        #     new_target = target.float()#.cpu()
        #     epoch_true = prev_batch[0].float().cpu().detach()
        #     epoch_pred = prev_batch[1][:, 1].cpu().detach()
        #     loss = self.roc_star.roc_star_loss(new_target, new_input, epoch_true, epoch_pred)
        #     self.roc_star.epoch_update_gamma(new_target.cpu().detach(), new_input.cpu().detach())
        #     return loss 

        if score_type == 'focal' or (score_type == 'roc-star' and epoch_num == 0):
            reduction = "sum"
            new_input = input[:, 1].cpu()
            new_target = target.float().cpu()
            p = new_input
            ce_loss = F.binary_cross_entropy(new_input, new_target).cuda() 
            loss = ce_loss
            p_t = (p * new_target + (1 - p) * (1 - new_target)).cuda()
            loss = ce_loss * ((1 - p_t) ** self.gamma)

            if self.alpha >= 0:
                alpha_t = (self.alpha * new_target + (1 - self.alpha) * (1 - new_target)).cuda()
                loss = alpha_t * loss

            if reduction == "mean":
                loss = loss.mean().cuda()
            elif reduction == "sum":
                loss = loss.sum().cuda()
            #self.roc_star.epoch_update_gamma(new_target.cpu().detach(), new_input.cpu().detach())
            return loss

        #Taken from https://github.com/fursovia/self-adj-dice/blob/master/sadice/loss.py
        if score_type == 'sadie':
            reduction = "sum"
            new_target = target.float().cpu()
            probs = torch.gather(input.cpu(), dim=1, index=new_target.type(torch.int64).unsqueeze(1))

            probs_with_factor = ((1 - probs) ** self.alpha) * probs
            loss = 1 - (2 * probs_with_factor + self.gamma) / (probs_with_factor + 1 + self.gamma)

            if reduction == "mean":
                return loss.mean()
            elif reduction == "sum":
                return loss.sum()
            elif reduction == "none" or reduction is None:
                return loss
            else:
                raise NotImplementedError(f"Reduction `{reduction}` is not supported.")
        
        new_input = input[:, 1].cuda()
        new_target = target.float().cuda()
        TP = torch.sum(new_input * new_target)
        FP = torch.sum(new_input * (1 - new_target))
        FN = torch.sum((1 - new_input) * new_target)
        if score_type == 'f1':
            loss = TP / (TP + 0.5 * (FP + FN))
        if score_type == 'fms': #fowlkes_mallows_score
            loss = TP / torch.sqrt((TP+FP) * (TP+FN) )
        return loss




class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class L1Loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(L1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, torch.squeeze(target, dim = 0), reduction=self.reduction )