from typing import Optional, Callable, List, Sequence, Any
from functools import partial
import math
import copy

import flax.linen as nn
import jax
import jax.numpy as jnp

from torch.hub import load_state_dict_from_url
from flax_utils import convert_pytorch_state_dict_to_flax
from flax.traverse_util import flatten_dict, unflatten_dict


model_urls = {
    # Weights ported from https://github.com/rwightman/pytorch-image-models/
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
    "efficientnet_b2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
    "efficientnet_b3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
    "efficientnet_b4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
    # Weights ported from https://github.com/lukemelas/EfficientNet-PyTorch/
    "efficientnet_b5": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
    "efficientnet_b6": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
    "efficientnet_b7": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def stochastic_depth(x, p: float, mode: str, key, training: bool = True):
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f'Drop probability has to be between 0 and 1, but got {p}')
    if mode not in ['batch', 'row']:
        raise ValueError(f'mode has to be either "batch" or "row", but got {mode}')
    if not training or p == 0.0:
        return x
    
    survival_rate = 1.0 - p
    if mode == 'row':
        size = [x.shape[0]] + [1] * (x.ndim - 1)
    else:
        size = [1] * x.ndim
    noise = jax.random.bernoulli(key=key, p=survival_rate, shape=size)
    if survival_rate > 0.0:
        noise.__truediv__(survival_rate)
    
    return x * noise
    

class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float,
        depth_mult: float,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"expand_ratio={self.expand_ratio}"
            f", kernel={self.kernel}"
            f", stride={self.stride}"
            f", input_channels={self.input_channels}"
            f", out_channels={self.out_channels}"
            f", num_layers={self.num_layers}"
            f")"
        )
        return s

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))



class SqueezeExcitation(nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """
    input_channels: int
    squeeze_channels: int
    activation: Callable = nn.relu
    scale_activation: Callable = nn.sigmoid

    @nn.compact
    def __call__(self, x):
        y = jnp.mean(x, axis=[1, 2], keepdims=True) # avg pooling
        y= nn.Conv(self.squeeze_channels, kernel_size=(1,1), name='fc1')(y)
        y = self.activation(y)
        y = nn.Conv(self.input_channels, kernel_size=(1,1), name='fc2')(y)
        y = self.scale_activation(y)
        return y * x

class ConvNormActivation(nn.Module):
    """
    Configurable block used for Convolution-Normalzation-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalzation-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in wich case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolutiuon layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optinal): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: Optional[int] =None
    groups: int = 1
    norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm
    activation_layer: Optional[Callable] = nn.relu
    dilation: int = 1
    bias: Optional[bool] = None

    @nn.compact
    def __call__(self, x, train=False):
        padding = self.padding
        bias = self.bias
        if padding is None:
            padding = (self.kernel_size - 1) // 2 * self.dilation
        if bias is None:
            bias = self.norm_layer is None
        
        x = nn.Conv(self.out_channels, kernel_size=(self.kernel_size, self.kernel_size), strides=self.stride, padding=padding, kernel_dilation=self.dilation, feature_group_count=self.groups, use_bias=bias, name='0')(x)

        if self.norm_layer is not None:
            x = self.norm_layer(name='1')(x, use_running_average=not train)
        if self.activation_layer is not None:
            x = self.activation_layer(x)
        
        return x


class MBConv(nn.Module):

    cfg: MBConvConfig
    stochastic_depth_prob: float
    norm_layer: Callable[..., nn.Module]
    se_layer: Callable[..., nn.Module] = SqueezeExcitation

    @nn.compact
    def __call__(self, x, train=False):
        if not (1 <= self.cfg.stride <= 2):
            raise ValueError('Illegal stride value')
        
        use_res_connect = self.cfg.stride == 1 and self.cfg.input_channels == self.cfg.out_channels
        activation_layer = nn.silu

        residual = x.copy()

        block_id = 0
        # Expand
        expanded_channels = self.cfg.adjust_channels(self.cfg.input_channels, self.cfg.expand_ratio)
        if expanded_channels != self.cfg.input_channels:
            x = ConvNormActivation(expanded_channels, kernel_size=1, norm_layer=self.norm_layer, activation_layer=activation_layer, name=f'block_{block_id}')(x)
            block_id += 1
        # Depthwise
        x = ConvNormActivation(expanded_channels, kernel_size=self.cfg.kernel, stride=self.cfg.stride,groups=expanded_channels, norm_layer=self.norm_layer, activation_layer=activation_layer, name=f'block_{block_id}')(x)

        # Squeeze and Excitation
        squeeze_channels = max(1, self.cfg.input_channels // 4)
        x = self.se_layer(expanded_channels, squeeze_channels, activation=nn.silu, name=f'block_{block_id+1}')(x)

        # Project
        x = ConvNormActivation(self.cfg.out_channels, kernel_size=1, norm_layer=self.norm_layer, activation_layer=None, name=f'block_{block_id+2}')(x)


        if use_res_connect:
            x = stochastic_depth(x, self.stochastic_depth_prob, 'row', self.make_rng('stochastic_depth'), training=train)
            x += residual
        
        return x


class EfficientNet(nn.Module):
    """
    EfficientNet main class

    Args:
        inverted_residual_setting (List[MBConvConfig]): Network structure
        dropout (float): The droupout probability
        stochastic_depth_prob (float): The stochastic depth probability
        num_classes (int): Number of classes
        block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
        norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
    """
    inverted_residual_setting: List[MBConvConfig]
    dropout: float
    stochastic_depth_prob: float = 0.2
    num_classes: int = 1000
    block: Optional[Callable[..., nn.Module]] = MBConv
    norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm

    @nn.compact
    def __call__(self, x, train=True):
        if not self.inverted_residual_setting:
            raise ValueError('The inverted_residual_setting should not be empty')
        elif not (
            isinstance(self.inverted_residual_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in self.inverted_residual_setting])
        ):
            raise TypeError('The inverted_residual_setting shoudl be List[MBConvConfig]')

        # First Layer
        firstconv_output_channels = self.inverted_residual_setting[0].input_channels

        y = ConvNormActivation(out_channels=firstconv_output_channels, kernel_size=3, stride=2, norm_layer=self.norm_layer, activation_layer=nn.silu, name='features_0')(x, train=train)

        # Inverted Residual Blocks
        total_stage_blocks = sum(cfg.num_layers for cfg in self.inverted_residual_setting)
        stage_block_id = 0

        for j, cfg in enumerate(self.inverted_residual_setting):
            for i in range(cfg.num_layers):
                block_cfg = copy.copy(cfg)
                if i > 0:
                    block_cfg.input_channels = block_cfg.out_channels
                    block_cfg.stride = 1
                
                sd_prob = self.stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                y = self.block(block_cfg, sd_prob, self.norm_layer, name=f'mbconv_{j+1}_{i}')(y)
                stage_block_id += 1

        # Building Last Layers
        lastconv_output_channels = 4 * self.inverted_residual_setting[-1].out_channels

        y = ConvNormActivation(lastconv_output_channels, kernel_size=1, norm_layer=self.norm_layer, activation_layer=nn.silu, name='features_8')(y)

        y = jnp.mean(y, axis=[1, 2])

        y = nn.Dropout(self.dropout, deterministic=not train)(y)
        y = nn.Dense(self.num_classes, name='classifier_1')(y)

        return y

def convert_pytorch_weights_to_flax(url, flax_model):
    state_dict = load_state_dict_from_url(url)
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), num=3)
    flax_state_dict = convert_pytorch_state_dict_to_flax(state_dict, flax_model, rngs={'params': key1, 'dropout': key2, 'stochastic_depth': key3})

    flattened_state_dict = flatten_dict(flax_state_dict)
    new_state_dict = {}
    bn_kernels = set()
    for key in flattened_state_dict.keys():
        new_key = copy.copy(key)        
        layer, i = key[0].split('_')
        if layer == 'features' and i not in ['0', '8']:
            new_key = (key[0].replace('features', 'mbconv') + '_' + key[1], ) + key[2:]
        
        if new_key[-1] in ['running_mean', 'running_var', 'num_batches_tracked']:
            new_key = ('batch_stats', ) + new_key[:-1] + (new_key[-1].split('_')[-1],)
            bn_kernels.add(('params',) + new_key[1:-1] + ('scale',))
        else:
            new_key = ('params', ) + new_key
        new_state_dict[new_key] = flattened_state_dict[key]

    for kernel in bn_kernels:
        new_state_dict[kernel] = new_state_dict.pop(kernel[:-1] + ('kernel',))
    new_state_dict = unflatten_dict(new_state_dict)
    return new_state_dict

def _efficientnet(
    arch: str,
    width_mult: float,
    depth_mult: float,
    dropout: float,
    pretrained: bool,
    progress: bool,
    **kwargs
):
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if pretrained:
        model = EfficientNet(inverted_residual_setting, dropout, **kwargs)
        params = convert_pytorch_weights_to_flax(model_urls[arch], model)
    else:
        model = EfficientNet(inverted_residual_setting, dropout, **kwargs)
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), num=3)
        params = model.init({'params': key1, 'dropout': key2, 'stochastic_depth': key3}, jnp.ones(1, 224, 224, 3)) # Initialization call
    
    return model, params




def efficientnet_b0(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    """
    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b0", 1.0, 1.0, 0.2, pretrained, progress, **kwargs)


def efficientnet_b1(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    """
    Constructs a EfficientNet B1 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b1", 1.0, 1.1, 0.2, pretrained, progress, **kwargs)


def efficientnet_b2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B2 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b2", 1.1, 1.2, 0.3, pretrained, progress, **kwargs)


def efficientnet_b3(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    """
    Constructs a EfficientNet B3 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b3", 1.2, 1.4, 0.3, pretrained, progress, **kwargs)


def efficientnet_b4(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    """
    Constructs a EfficientNet B4 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b4", 1.4, 1.8, 0.4, pretrained, progress, **kwargs)


def efficientnet_b5(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    """
    Constructs a EfficientNet B5 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(
        "efficientnet_b5",
        1.6,
        2.2,
        0.4,
        pretrained,
        progress,
        norm_layer=partial(nn.BatchNorm, epsilon=0.001, momentum=1-0.01),
        **kwargs,
    )


def efficientnet_b6(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    """
    Constructs a EfficientNet B6 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(
        "efficientnet_b6",
        1.8,
        2.6,
        0.5,
        pretrained,
        progress,
        norm_layer=partial(nn.BatchNorm, epsilon=0.001, momentum=1-0.01),
        **kwargs,
    )


def efficientnet_b7(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    """
    Constructs a EfficientNet B7 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(
        "efficientnet_b7",
        2.0,
        3.1,
        0.5,
        pretrained,
        progress,
        norm_layer=partial(nn.BatchNorm, epsilon=0.001, momentum=1-0.01),
        **kwargs,
    )
