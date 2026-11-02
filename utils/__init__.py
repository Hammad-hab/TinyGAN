import torch

def get_device():
    if torch.cuda.is_available():
        print('[DEVICE] Discovered CUDA')
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print('[DEVICE] Discovered MPS')
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def fdGetNoisy(image, time, ns,):
    eps = torch.rand_like(image, )
    alpha = ns._alphab[time].view(-1, 1, 1, 1)
    xt = torch.sqrt(alpha)*image + torch.sqrt(1-alpha)*eps
    return xt, eps
