#from tkinter import E
from torch.optim import Optimizer
import torch
import math
from skorch.callbacks import LRScheduler
import hyper_params
from torch.optim.lr_scheduler import LinearLR
from typing import Callable, Iterable, Tuple
from torch.distributions.bernoulli import Bernoulli

class mAdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
        local_normalization (bool): 
        max_grad_norm (bool): 
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0, correct_bias=True,
                 local_normalization=False, max_grad_norm=-1, warmup = hyper_params.NUM_WORMUP_STEPS):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias,
                        local_normalization=local_normalization, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)
        self.warmup = warmup
        self._step = 0

    def rate(self, step = None):
        "Implement `lrate` above"
        if step < self.warmup:
            return float(step) / float(max(1.0, self.warmup))
        elif step == self.warmup:
            return 1.0
        return 1.0 #(1 / (1 - hyper_params.WARMUP_PRECENTAGE)) - (step / hyper_params.NUM_LEFT_STEPS)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        updates = []

        self._step += 1
        rate = self.rate(self._step)

        for group in self.param_groups:

            group_updates = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Normalize gradients locally (layer-wise)
                if group["local_normalization"]:
                    torch.nn.utils.clip_grad_norm_(p, group["max_grad_norm"])

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = rate * group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

                group_updates.append((exp_avg, denom))

            updates.append(group_updates)

        return loss, updates


class myLRScheduler(LRScheduler):
    #Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
    def _get_scheduler(self, net, policy, ft_epochs, last_epoch=-1, num_warmup_steps = hyper_params.NUM_WORMUP_STEPS, start_factor=1.0 / 3, end_factor=1.0,
                                   **scheduler_kwargs):
        return LinearLR(net.optimizer_, last_epoch = last_epoch)


class ChildTuningAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = hyper_params.WEIGHT_DECAY,
        correct_bias: bool = True,
        reserve_p = hyper_params.RESERVE_P,
        mode = None,
        warmup = hyper_params.NUM_WORMUP_STEPS,
        warmup_precentage = hyper_params.WARMUP_PRECENTAGE,
        num_left_steps = hyper_params.NUM_LEFT_STEPS,
        model_size = 201, 
        bert_lr: float = 0,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, bert_lr = bert_lr)
        super().__init__(params, defaults)

        self.gradient_mask = None
        self.reserve_p = reserve_p
        self.mode = mode
        self.warmup = warmup
        self._step = 0
        self.model_size = model_size
        self.lrs = []
        self.num_left_steps = num_left_steps


        for group in self.param_groups:
            self.lrs.append(group['lr'])
        self.warmup_precentage = warmup_precentage

    def rate(self, step = None):
        "Implement `lrate` above"
        if not hyper_params.DO_WARMUP:
            return 1.0 / (math.pow(2, step))
        if step < self.warmup:
            return float(step) / float(max(1.0, self.warmup))
        elif step == self.warmup:
            return 1.0
        return (1 / (1 - self.warmup_precentage)) - (step / self.num_left_steps)

    def set_gradient_mask(self, gradient_mask):
        self.gradient_mask = gradient_mask

    def have_gradient_mask(self):
        return self.gradient_mask is None

    def step(self, closure: Callable = None, **fit_params):

        rate = 1 #self._step
        if hyper_params.PERFORM_WARMPUP:
            self._step += 1
            rate = self.rate(self._step)

        """
        Performs a single optimization step.
        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure(**fit_params)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # =================== HACK BEGIN =======================         
                if group['mode'] is not None:
                    if group['mode'] == 'ChildTuning-D':
                        if p in self.gradient_mask:
                            grad *= self.gradient_mask[p].cuda()
                    else: 
                        # ChildTuning-F
                        grad_mask = Bernoulli(grad.new_full(size=grad.size(), fill_value=self.reserve_p))
                        grad *= grad_mask.sample() / self.reserve_p
                # =================== HACK END =======================

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                lr = group["lr"] if ("is_bert" not in group or not group["is_bert"]) else group["bert_lr"]
 
                step_size = rate * lr
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                p.data.add_(p.data, alpha=-lr * group["weight_decay"])

        return loss