import torch
def compute_erank(matrix, *args, **kwargs):
    if matrix.dtype == torch.float16:
        matrix = matrix.to(torch.float32)
    U, S, V = torch.svd(matrix)
    p = S / torch.norm(S, p=1)
    entropy  = -torch.sum(p * torch.log(p + 1e-10))
    erank = torch.exp(entropy)
    return erank

