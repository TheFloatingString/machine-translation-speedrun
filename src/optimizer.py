import torch

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power (orthogonalization) of a matrix G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(torch.float32)
    X /= (X.norm() + eps) # normalize
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.9, nsr_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nsr_steps=nsr_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = zeropower_via_newtonschulz5(buf, steps=group['nsr_steps'])
                g *= (max(p.size(0), p.size(1)) ** 0.5) # scale by spectral radius of random matrix
                p.data.add_(g, alpha=-lr)
        return loss

class CombinedOptimizer(torch.optim.Optimizer):
    """
    A simple wrapper to step multiple optimizers together.
    """
    def __init__(self, optimizers):
        # We need to call the super init to set up internal hooks
        # We pass the combined param groups from all optimizers
        combined_params = []
        for opt in optimizers:
            combined_params.extend(opt.param_groups)
        
        super().__init__(combined_params, {})
        self.optimizers = optimizers

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dict):
        for opt, state in zip(self.optimizers, state_dict):
            opt.load_state_dict(state)
