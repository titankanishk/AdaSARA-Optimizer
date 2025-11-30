import torch

class AdaSARA(torch.optim.Optimizer):
    """
    AdaSARA Optimizer
    Adaptive Speed-Aware Repulsive Adjustment:
    Adds a velocity-aware correction to Adam's second moment term
    to improve stability in rapidly changing gradient landscapes.
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, G=1e-3):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, G=G)
        super(AdaSARA, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:

                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['delta'] = torch.zeros_like(p)

                m, v, delta = state['m'], state['v'], state['delta']
                beta1, beta2 = group['beta1'], group['beta2']
                eps, G = group['eps'], group['G']
                lr = group['lr']

                # First moment
                m.mul_(beta1).add_(grad, alpha=(1 - beta1))

                # Baseline Adam second moment
                v_adam = beta2 * v + (1 - beta2) * grad * grad

                # Speed-aware interaction factor
                s = (torch.abs(delta) * torch.abs(grad)) / (
                    torch.abs(delta) + torch.abs(grad) + eps
                )

                # Apply repulsive correction
                v = v_adam + G * s

                # Bias correction
                m_hat = m / (1 - beta1)
                v_hat = v / (1 - beta2)

                # Update delta and parameters
                delta = -lr * m_hat / (torch.sqrt(v_hat) + eps)
                p.add_(delta)

                state['m'], state['v'], state['delta'] = m, v, delta

        return loss
