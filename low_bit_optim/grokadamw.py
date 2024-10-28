# Implementation logic is inspired from https://github.com/pytorch/ao/blob/main/torchao/prototype/low_bit_optim/adam.py

import math
import torch
from typing import Optional
from torch.optim import Optimizer
from torch.cuda.amp import autocast
from typing import Iterable, Callable, Optional
from torch import Tensor
from torch.distributed._tensor import DTensor
from torchao.prototype.low_bit_optim.subclass_8bit import OptimState8bit

class _GrokAdamWBase(Optimizer):
    def __init__(self, 
                 params: Iterable[torch.Tensor], 
                 lr: float = 1e-3, 
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, 
                 weight_decay: float = 1e-2, 
                 block_size: int = 256,
                 alpha_init: float = 0.98, 
                 lamb: float = 2.0,
                 gamma: float = 0.1, 
                 grokking_signal_fns: Optional[list[Callable[[], float]]] = None,
                 grokking_signal_decay_rate: float = 0.1, 
                 gradient_clipping: float = 1.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha_init <= 1.0:
            raise ValueError(f"Invalid alpha_init value: {alpha_init}")

        self.block_size = block_size
        if grokking_signal_fns is not None:
            if not isinstance(grokking_signal_fns, list):
                grokking_signal_fns = [grokking_signal_fns]

        defaults = dict(lr=lr, 
                        betas=betas, 
                        eps=eps, 
                        weight_decay=weight_decay,
                        alpha_init=alpha_init, 
                        lamb=lamb, 
                        gamma=gamma,
                        grokking_signal_fns=grokking_signal_fns,
                        grokking_signal_decay_rate=grokking_signal_decay_rate,
                        gradient_clipping=gradient_clipping)
        super(_GrokAdamWBase, self).__init__(params, defaults)

    def state_dict(self):
        state_dict = super().state_dict()
        for group in state_dict['param_groups']:
            group['grokking_signal_fns'] = None  # Cannot serialize functions
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            group['grokking_signal_fns'] = self.defaults['grokking_signal_fns']

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('grokking_signal_fns', [])
            group.setdefault('grokking_signal_decay_rate', 0.1)
            group.setdefault('gradient_clipping', 1.0)
    
    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool):
        raise NotImplementedError

    def _new_buffer(self, p: Tensor, signed: bool):
        if p.numel() >= 4096 and p.numel() % self.block_size == 0:
            if isinstance(p, DTensor):
                out = DTensor.from_local(
                    local_tensor=self._subclass_zeros(p.to_local(), signed, self.block_size),
                    device_mesh=p.device_mesh,
                    placements=p.placements,
                    run_check=False,
                )
            else:
                out = self._subclass_zeros(p, signed, self.block_size)
        else:
            out = torch.zeros_like(p)
        return out
    
    @staticmethod
    def _default_grokking_signal(train_loss: Optional[float], eval_loss: Optional[float]) -> float:
        """Default grokking signal function based on loss difference."""
        if train_loss is None or eval_loss is None:
            return 0.0
        diff = max(0, eval_loss - train_loss)
        max_loss = max(eval_loss, train_loss)
        return diff / max_loss if max_loss > 0 else 0.0

    def _compute_grokking_signal(self, group: dict) -> Optional[float]:
        """Computes a combined grokking signal from multiple functions."""
        if group['grokking_signal_fns'] is None:
            train_loss = group.get('train_loss', None)
            eval_loss = group.get('eval_loss', None)
            return self._default_grokking_signal(train_loss, eval_loss)

        signals = []
        for fn in group['grokking_signal_fns']:
            try:
                signal = fn()
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                raise NotImplementedError(f"Error in grokking_signal_fn: {e}. Ignoring this function.")
        
        return sum(signals) / len(signals) if signals else None
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        return self._step_impl(closure)
    
    def _step_impl(self, closure: Optional[Callable[[], float]]) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with torch._dynamo.utils.disable_cache_limit():
            for group in self.param_groups:
                grokking_signal = self._compute_grokking_signal(group)
                for p in group["params"]:
                    if p.grad is None:
                        continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradient is not supported")
                
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = self._new_buffer(p, True)
                    state["exp_avg_sq"] = self._new_buffer(p, False)
                    state["grok_ema"] = self._new_buffer(p, False)
                
                state["step"] += 1

                if not isinstance(group["lr"], Tensor):
                    raise RuntimeError(
                        "lr was changed to a non-Tensor object. If you want to update lr, please use "
                        "optim.param_groups[0]['lr'].fill_(new_lr)"
                    )
                if group["gradient_clipping"] > 0:
                    torch.nn.utils.clip_grad_norm_(p, group["gradient_clipping"])
                
                torch.compile(single_param_grokadam, fullgraph = True, dynamic = False)(
                    p,
                    grad,
                    grokking_signal,
                    state["step"],
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["grok_ema"],
                    group["lamb"],
                    group["alpha_init"],
                    group["gamma"],
                    group["lr"],
                    group["betas"][0],
                    group["betas"][1],
                    group["weight_decay"],
                    group["grokking_signal_decay_rate"],
                    group["eps"],
                )

        return loss

def single_param_grokadam(
    p: Tensor,
    grad: Tensor,
    grokking_signal: Tensor,
    step: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    grok_ema: Tensor,
    lamb: Tensor,
    alpha_init: Tensor,
    gamma: Tensor,
    lr: Tensor,
    beta1: float,
    beta2: float,
    weight_decay: float,
    grokking_signal_decay_rate: float,
    eps: float,
):
    grad_f32 = grad.float()

    grok_ema_fp32 = grok_ema.float()
    exp_avg_f32 = exp_avg.float()
    exp_avg_sq_f32 = exp_avg_sq.float()

    exp_avg.copy_(exp_avg_f32)
    exp_avg_sq.copy_(exp_avg_sq_f32)

    layer_beta1 = beta1 * (1 - gamma)**step
    if grokking_signal is not None:
        alpha_init = alpha_init * math.exp(-grokking_signal_decay_rate * grokking_signal)
    grok_ema_fp32.mul_(alpha_init).add_(grad, alpha = 1 - alpha_init)
    grok_grad_fp32 = grad_f32 + lamb * grok_ema_fp32

    # Update moments
    exp_avg_f32.mul_(layer_beta1).add_(grok_grad_fp32, alpha = 1 - layer_beta1)
    exp_avg_sq_f32.mul_(beta2).addcmul_(grok_grad_fp32, grok_grad_fp32, value= 1 - beta2)

    # Bias correction
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    
    step_size = lr * ((bias_correction2) ** 0.5) / bias_correction1

    # Update parameters
    p.mul_(1 - lr * weight_decay)
    p.addcdiv_(exp_avg_f32, exp_avg_sq_f32.sqrt().add_(eps), value=-step_size)

class GrokAdamw8bit(_GrokAdamWBase):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        alpha_init=0.98,
        lamb=2.0,
        gamma=0.1,
        *,
        block_size=256,
        grokking_signal_fns = None,
        grokking_signal_decay_rate = 0.1,
        gradient_clipping=1.0
    ) -> None:
        super().__init__(
            params, 
            lr, 
            betas, 
            eps, 
            weight_decay, 
            block_size,
            alpha_init,
            lamb,
            gamma,
            grokking_signal_fns,
            grokking_signal_decay_rate,
            gradient_clipping,
        )

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimState8bit.zeros(p.shape, signed, block_size, p.device)