import pytest
import torch

from torchlars._adaptive_lr import compute_adaptive_lr


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_compare_cpu_and_gpu(dtype):
    param_norm = torch.tensor(1., dtype=dtype)
    grad_norm = torch.tensor(1., dtype=dtype)
    adaptive_lr_cpu = torch.tensor(0., dtype=dtype)

    weight_decay = 1.
    eps = 2.
    trust_coef = 1.

    adaptive_lr_cpu = compute_adaptive_lr(
        param_norm,
        grad_norm,
        weight_decay,
        eps,
        trust_coef,
        adaptive_lr_cpu)

    param_norm = torch.tensor(1., dtype=dtype, device='cuda')
    grad_norm = torch.tensor(1., dtype=dtype, device='cuda')
    adaptive_lr_gpu = torch.tensor(0., dtype=dtype, device='cuda')

    weight_decay = 1.
    eps = 2.
    trust_coef = 1.

    adaptive_lr_gpu = compute_adaptive_lr(
        param_norm,
        grad_norm,
        weight_decay,
        eps,
        trust_coef,
        adaptive_lr_gpu)

    assert torch.allclose(adaptive_lr_cpu, adaptive_lr_gpu.cpu())