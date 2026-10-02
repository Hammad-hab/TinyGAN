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
