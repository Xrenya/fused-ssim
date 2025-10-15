from typing import NamedTuple
import torch.nn as nn
import torch
import warnings

if torch.cuda.is_available():
    from fused_ssim_cuda import fusedssim, fusedssim_backward
else:
    # Fallback – you can raise an error or provide a pure‑PyTorch fallback
    raise RuntimeError("CUDA is not found! MPS/XPU implementation is not implemented")
#TODO: add mask for the rest of the implementations
# elif torch.mps.is_available():
#     from fused_ssim_mps import fusedssim, fusedssim_backward
# elif hasattr(torch, 'xpu') and torch.xpu.is_available():
#     from fused_ssim_xpu import fusedssim, fusedssim_backward


allowed_padding = ["same", "valid"]

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, mask=None, padding="same", train=True):
        """
        Forward pass of the fused SSIM map.

        Args:
            ctx: context for backward
            C1, C2: SSIM constants
            img1, img2: input tensors of shape (N, C, H, W)
            mask: optional mask of shape (N, 1, H, W) or (N, H, W) or (1, H, W)
                  valid = 1, invalid = 0
            padding: "same" or "valid"
            train: flag that is passed to the fused kernel
        """
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, mask, train)

        if padding == "valid":
            ssim_map = ssim_map[:, :, 5:-5, 5:-5]
            if mask is not None:
                mask = mask[:, :, 5:-5, 5:-5]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, mask)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return ssim_map
        # TODO: it is already `mean()` -> the function should not call `mean()`

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, mask = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding

        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
        # mask passed to zeros gradients in correspoding positions
        grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, mask, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, grad, None, None, None

def fused_ssim(img1, img2, mask=None, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    img1 = img1.contiguous()
    map = FusedSSIMMap.apply(C1, C2, img1, img2, mask, padding, train)
    return map
