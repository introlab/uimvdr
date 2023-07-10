import torch

def sqrt_hann_window(
    window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False
):
    return torch.sqrt(
        torch.hann_window(
            window_length, periodic=periodic, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )
    )

def cuda_sqrt_hann_window(
    window_length, periodic=True, dtype=None, layout=torch.strided, device="cuda", requires_grad=False
):
    return torch.sqrt(
        torch.hann_window(
            window_length, periodic=periodic, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )
    )