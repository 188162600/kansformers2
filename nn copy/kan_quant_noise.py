# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from .kan_layer import KANLayer
def kan_quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module,KANLayer)


    assert (
        module.weight.size(0) % block_size == 0
    ), "Input features must be a multiple of block sizes"


    def _forward_pre_hook(mod, input):
        # no noise for evaluation
       
    
        # gather weight and sizes
        weight = mod.weight
        in_features = weight.size(0)* weight.size(1)
        out_features =weight.size(2)

        # split weight matrix into blocks and randomly drop selected blocks
        mask = torch.zeros(
            in_features // block_size * out_features, device=weight.device
        )
        mask.bernoulli_(p)
        mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
        mask=mask.transpose(0,1).reshape(weight.size(0),weight.size(1),weight.size(2))
        
        # scale weights and apply mask
        mask = mask.to(
            torch.bool
        )  # x.bool() is not currently supported in TorchScript
        s = 1 / (1 - p)
        mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module
