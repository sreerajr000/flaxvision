"""Flax implementation of ResNet"""


from functools import partial
from typing import Any, Callable, Sequence, Tuple, List, Type, Union

import jax
from flax import linen as nn
import jax.numpy as jnp

from torch.hub import load_state_dict_from_url
from flax_utils import convert_pytorch_state_dict_to_flax
from flax.traverse_util import flatten_dict, unflatten_dict

ModuleDef = Any

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


class ResNetBlock(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)
    base_width: int = 64
    groups: int = 1

    @nn.compact
    def __call__(self, x,):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides, padding=((1,1),(1,1)), name='conv1')(x)
        y = self.norm(name='bn1')(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3),  padding=((1,1),(1,1)), name='conv2')(y)
        y = self.norm(scale_init=nn.initializers.zeros, name='bn2')(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1),
                                self.strides,  padding=((0,0),(0,0)), name='downsample_0')(residual)
            residual = self.norm(name='downsample_1')(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)
    expansion: int = 4
    base_width: int = 64
    groups: int = 1

    @nn.compact
    def __call__(self, x):
        width = int(self.filters * (self.base_width / 64.0)) * self.groups

        residual = x
        y = self.conv(width, (1, 1), padding=((0,0),(0,0)), name='conv1')(x)
        y = self.norm(name='bn1')(y)
        y = self.act(y)
        y = self.conv(width, (3, 3), self.strides, padding=((1,1),(1,1)), feature_group_count=self.groups, name='conv2')(y)
        y = self.norm(name='bn2')(y)
        y = self.act(y)
        y = self.conv(self.filters * self.expansion, (1, 1), padding=((0,0),(0,0)), name='conv3')(y)
        y = self.norm(name='bn3', scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * self.expansion, (1, 1),
                                self.strides, padding=((0,0),(0,0)), name='downsample_0')(residual)
            residual = self.norm(name='downsample_1')(residual)

        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNetV1."""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv
    base_width: int = 64
    groups: int = 1
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.BatchNorm,
                    use_running_average=not train,
                    momentum=0.9,
                    epsilon=1e-5,
                    dtype=self.dtype)

        x = conv(self.num_filters, (7, 7), (2, 2),
                padding=[(3, 3), (3, 3)],
                name='conv1')(x)
        x = norm(name='bn1')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1,1),(1,1)))
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2 ** i,
                                strides=strides,
                                conv=conv,
                                norm=norm,
                                base_width=self.base_width,
                                groups=self.groups,
                                act=self.act,
                                name=f'layer{i+1}_{j}')(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype, name='fc')(x)
        x = jnp.asarray(x, self.dtype)
        return x


def convert_pytorch_weights_to_flax(url, flax_model):
    state_dict = load_state_dict_from_url(url)
    flax_state_dict = convert_pytorch_state_dict_to_flax(state_dict, flax_model)

    flax_state_dict = flatten_dict(flax_state_dict)
    updated_params = {}
    for keys in flax_state_dict.keys():
        keys_copy = list(keys)
        if keys_copy[-1] in ['running_mean', 'running_var']:
            # batch_stats
            keys_copy[-1] = keys_copy[-1].split('_')[-1]
            keys_copy = ['batch_stats'] + keys_copy
        else:
            # params
            if ('bn' in keys_copy[-2] or 'downsample_1' == keys_copy[-2]) and keys_copy[-1] == 'kernel':
                keys_copy[-1] = 'scale'
            keys_copy = ['params'] + keys_copy

        updated_params[tuple(keys_copy)] = flax_state_dict[keys]
    # print(updated_params.keys())
    flax_state_dict = unflatten_dict(updated_params)
    del updated_params
    return flax_state_dict


def _resnet(
    arch: str,
    block: Type[Union[ResNetBlock, BottleneckResNetBlock]],
    layers: List[int],
    pretrained: bool,
    num_classes,
    base_width=64,
    groups=1
):
    
    if pretrained:
        model = ResNet(block_cls=block, stage_sizes=layers, num_classes=1000, base_width=base_width, groups=groups)
        params = convert_pytorch_weights_to_flax(model_urls[arch], model)
    else:
        model = ResNet(block_cls=block, stage_sizes=layers, num_classes=num_classes, base_width=base_width, groups=groups)
        key1, key2 = jax.random.split(jax.random.PRNGKey(0))
        x = jax.random.normal(key1, (1, 224, 224, 3)) # Dummy input data
        params = model.init(key2, x) # Initialization call
    return model, params


def resnet18(pretrained: bool = False, num_classes=1000):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): default 1000
    """
    return _resnet("resnet18", ResNetBlock, [2, 2, 2, 2], pretrained, num_classes)


def resnet34(pretrained: bool = False, num_classes=1000):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): default 1000
    """
    return _resnet("resnet34", ResNetBlock, [3, 4, 6, 3], pretrained, num_classes)


def resnet50(pretrained: bool = False, num_classes=1000):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): default 1000
    """
    return _resnet("resnet50", BottleneckResNetBlock, [3, 4, 6, 3], pretrained, num_classes)


def resnet101(pretrained: bool = False, num_classes=1000):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): default 1000
    """
    return _resnet("resnet101", BottleneckResNetBlock, [3, 4, 23, 3], pretrained, num_classes)


def resnet152(pretrained: bool = False, num_classes=1000):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): default 1000
    """
    return _resnet("resnet152", BottleneckResNetBlock, [3, 8, 36, 3], pretrained, num_classes)


def resnext50_32x4d(pretrained: bool = False, num_classes=1000):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): default 1000
    """
    return _resnet("resnext50_32x4d", BottleneckResNetBlock, [3, 4, 6, 3], pretrained, num_classes=num_classes, base_width=4, groups=32)


def resnext101_32x8d(pretrained: bool = False, num_classes=1000):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): default 1000
    """
    return _resnet("resnext101_32x8d", BottleneckResNetBlock, [3, 4, 23, 3], pretrained, num_classes=num_classes, base_width=8, groups=32)


def wide_resnet50_2(pretrained: bool = False, num_classes=1000):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): default 1000
    """
    return _resnet("wide_resnet50_2", BottleneckResNetBlock, [3, 4, 6, 3], pretrained, num_classes=num_classes, base_width=64*2)


def wide_resnet101_2(pretrained: bool = False, num_classes=1000):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): default 1000
    """
    return _resnet("wide_resnet101_2", BottleneckResNetBlock, [3, 4, 23, 3], pretrained, num_classes=num_classes, base_width=64*2)
