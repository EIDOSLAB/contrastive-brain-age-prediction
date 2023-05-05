# -*- coding: utf-8 -*-
##########################################################################
# Code version 6207dffcc20f461bdb742f5d8a2f6641483b9d83
##########################################################################


"""
Each solution to be tested should be stored in its own directory within
submissions/. The name of this new directory will serve as the ID for
the submission. If you wish to launch a RAMP challenge you will need to
provide an example solution within submissions/starting_kit/. Even if
you are not launching a RAMP challenge on RAMP Studio, it is useful to
have an example submission as it shows which files are required, how they
need to be named and how each file should be structured.
"""

# Filename: estimator_mse.py
# Run id: 
# 
import os
ARCHITECTURE = os.environ.get("ARCHITECTURE", "resnet18")


from collections import OrderedDict
from abc import ABCMeta
import progressbar
import nibabel
import numpy as np
from nilearn.masking import unmask
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torchvision import transforms
import math

############################################################################
# Define here some selectors
############################################################################

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """ Select only the requested data associatedd features from the the
    input buffered data.
    """
    MODALITIES = OrderedDict([
        ("vbm", {
            "shape": (1, 121, 145, 121),
            "size": 519945}),
        ("quasiraw", {
            "shape": (1, 182, 218, 182),
            "size": 1827095}),
        ("xhemi", {
            "shape": (8, 163842),
            "size": 1310736}),
        ("vbm_roi", {
            "shape": (1, 284),
            "size": 284}),
        ("desikan_roi", {
            "shape": (7, 68),
            "size": 476}),
        ("destrieux_roi", {
            "shape": (7, 148),
            "size": 1036})
    ])
    MASKS = {
        "vbm": {
            "path": None,
            "thr": 0.05},
        "quasiraw": {
            "path": None,
            "thr": 0}
    }

    def __init__(self, dtype, mock=False):
        """ Init class.
        Parameters
        ----------
        dtype: str
            the requested data: 'vbm', 'quasiraw', 'vbm_roi', 'desikan_roi',
            'destrieux_roi' or 'xhemi'.
        """
        if dtype not in self.MODALITIES:
            raise ValueError("Invalid input data type.")
        self.dtype = dtype

        data_types = list(self.MODALITIES.keys())
        index = data_types.index(dtype)
        
        cumsum = np.cumsum([item["size"] for item in self.MODALITIES.values()])
        
        if index > 0:
            self.start = cumsum[index - 1]
        else:
            self.start = 0
        self.stop = cumsum[index]
        
        self.masks = dict((key, val["path"]) for key, val in self.MASKS.items())
        self.masks["vbm"] = os.environ.get("VBM_MASK")
        self.masks["quasiraw"] = os.environ.get("QUASIRAW_MASK")

        self.mock = mock
        if mock:
            return

        for key in self.masks:
            if self.masks[key] is None or not os.path.isfile(self.masks[key]):
                raise ValueError("Impossible to find mask:", key, self.masks[key])
            arr = nibabel.load(self.masks[key]).get_fdata()
            thr = self.MASKS[key]["thr"]
            arr[arr <= thr] = 0
            arr[arr > thr] = 1
            self.masks[key] = nibabel.Nifti1Image(arr.astype(int), np.eye(4))

    def fit(self, X, y):
        return self

    def transform(self, X):
        if self.mock:
            #print("transforming", X.shape)
            data = X.reshape(self.MODALITIES[self.dtype]["shape"])
            #print("mock data:", data.shape)
            return data
        
        # print(X.shape)
        select_X = X[self.start:self.stop]
        if self.dtype in ("vbm", "quasiraw"):
            im = unmask(select_X, self.masks[self.dtype])
            select_X = im.get_fdata()
            select_X = select_X.transpose(2, 0, 1)
        select_X = select_X.reshape(self.MODALITIES[self.dtype]["shape"])
        return select_X

class Crop(object):
    """ Crop the given n-dimensional array either at a random location or
    centered.
    """
    def __init__(self, shape, type="center", keep_dim=False):
        assert type in ["center", "random"]
        self.shape = shape
        self.copping_type = type
        self.keep_dim = keep_dim

    def __call__(self, X):
        img_shape = np.array(X.shape)

        if type(self.shape) == int:
            size = [self.shape for _ in range(len(self.shape))]
        else:
            size = np.copy(self.shape)
        
        # print('img_shape:', img_shape, 'size', size)
        
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            
            if self.copping_type == "center":
                delta_before = int((img_shape[ndim] - size[ndim]) / 2.0)
            
            elif self.copping_type == "random":
                delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
            
            indexes.append(slice(delta_before, delta_before + size[ndim]))
        
        if self.keep_dim:
            mask = np.zeros(img_shape, dtype=np.bool)
            mask[tuple(indexes)] = True
            arr_copy = X.copy()
            arr_copy[~mask] = 0
            return arr_copy
        
        _X = X[tuple(indexes)]
        # print('cropped.shape', _X.shape)
        return _X

class Pad(object):
    """ Pad the given n-dimensional array
    """
    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.kwargs = kwargs

    def __call__(self, X):
        _X = self._apply_padding(X)
        return _X

    def _apply_padding(self, arr):
        orig_shape = arr.shape
        padding = []
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append([half_shape_i, half_shape_i])
            else:
                padding.append([half_shape_i, half_shape_i + 1])
        for cnt in range(len(arr.shape) - len(padding)):
            padding.append([0, 0])
        fill_arr = np.pad(arr, padding, **self.kwargs)
        return fill_arr

############################################################################
# Define here your dataset
############################################################################

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None, transforms=None, indices=None):
        self.T = transforms
        self.X = X
        self.y = y
        self.indices = indices
        if indices is None:
            self.indices = range(len(X))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_i = self.indices[i]
        x = self.X[real_i]

        if self.T is not None:
            x = self.T(x)

        if self.y is not None:
            y = self.y[real_i]
            return x, y
        else:
            return x


############################################################################
# Define here your regression model
############################################################################
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    Standard 3D-ResNet architecture with big initial 7x7x7 kernel.
    It can be turned in mode "classifier", outputting a vector of size <n_classes> or
    "encoder", outputting a latent vector of size 512 (independent of input size).
    Note: only a last FC layer is added on top of the "encoder" backbone.
    """
    def __init__(self, block, layers, in_channels=1,
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, initial_kernel_size=7):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.name = "resnet"
        self.inputs = None
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        initial_stride = 2 if initial_kernel_size==7 else 1
        padding = (initial_kernel_size-initial_stride+1)//2
        self.conv1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=initial_kernel_size, stride=initial_stride, padding=padding, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        channels = [64, 128, 256, 512]

        self.layer1 = self._make_layer(block, channels[0], layers[0])
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        return torch.flatten(x5, 1)

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2],  **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)    

model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}

class SupRegResNet(nn.Module):
    """encoder + regressor"""
    def __init__(self, name='resnet50'):
        super().__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, 1)

    def forward(self, x):
        return self.encoder(x)
        # return self.fc(self.encoder(x))

class AlexNet3D(nn.Module):
    def __init__(self):
        """
        :param num_classes: int, number of classes
        :param mode:  "classifier" or "encoder" (returning 128-d vector)
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool3d(1),
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        return x

class SupRegAlexNet(nn.Module):
    """encoder + regressor"""
    def __init__(self,):
        super().__init__()
        self.encoder = AlexNet3D()
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        feats = self.features(x)
        return feats
        # return self.fc(feats), feats
    
    def features(self, x):
        return self.encoder(x)

class DenseNet(nn.Module):
    """3D-Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        mode (str) - "classifier" or "encoder" (all but last FC layer)
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(3, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4, in_channels=1,
                 memory_efficient=False):
        super(DenseNet, self).__init__()
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.num_features = num_features


        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool3d(features, 1)
        out = torch.flatten(out, 1)
        return out.squeeze(dim=1)
       

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                memory_efficient=memory_efficient,
                )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


def _densenet(arch, growth_rate, block_config, num_init_features, **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    return model


def densenet121(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, **kwargs)

class SupRegDenseNet(nn.Module):
    """encoder + regressor"""
    def __init__(self,):
        super().__init__()
        self.encoder = densenet121()
        self.fc = nn.Linear(self.encoder.num_features, 1)

    def forward(self, x):
        feats = self.features(x)
        return feats
        # return self.fc(feats), feats
    
    def features(self, x):
        return self.encoder(x)
        
class RegressionModel(metaclass=ABCMeta):
    __model_local_weights__ = os.path.join(os.path.dirname(__file__), os.environ.get("MODEL", "weights.pth"))
    __metadata_local_weights__ = os.path.join(os.path.dirname(__file__), "metadata.pkl")

    def __init__(self, model, batch_size=15, transforms=None):
        self.model = model
        self.batch_size = batch_size
        self.transforms = transforms
        self.indices = None

    def fit(self, X, y):
        """ Restore weights.
        """        
        if not os.path.isfile(self.__model_local_weights__):
            raise ValueError("You must provide the model weigths in your submission folder.")
        state = torch.load(self.__model_local_weights__, map_location="cpu")

        if "model" not in state:
            raise ValueError("Model weigths are searched in the state dictionary at the 'model' key location.")
        self.model.load_state_dict(state["model"], strict=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        dataset = Dataset(X, transforms=self.transforms, indices=self.indices)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.model.eval()
        outputs = []

        with progressbar.ProgressBar(max_value=len(testloader)) as bar:
            for cnt, inputs in enumerate(testloader):
                inputs = inputs.float().to(device)
                # print("Batch size", inputs.shape)
                with torch.no_grad():
                    out = self.model(inputs)
                    # out = torch.randn((inputs.shape[0], 128))
                
                outputs.append(out.detach())
                bar.update(cnt)

        outputs = torch.cat(outputs, dim=0)
        return outputs.detach().cpu().numpy()


############################################################################
# Define here your estimator pipeline
############################################################################

def get_estimator(mock=False) -> Pipeline:
    """ Build your estimator here.
    Notes
    -----
    In order to minimize the memory load the first steps of the pipeline
    are applied directly as transforms attached to the Torch Dataset.
    Notes
    -----
    It is recommended to create an instance of sklearn.pipeline.Pipeline.
    """
    if "resnet" in ARCHITECTURE:
        net = SupRegResNet(ARCHITECTURE)
    elif ARCHITECTURE == "alexnet":
        net = SupRegAlexNet()
    elif "densenet" in ARCHITECTURE:
        net = SupRegDenseNet()

    selector = FeatureExtractor("vbm", mock=mock)
    preproc = transforms.Compose([
        transforms.Lambda(lambda x: selector.transform(x)),
        # Crop((1, 121, 128, 121), type="center"),
        # Pad((1, 128, 128, 128)),
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        transforms.Normalize(mean=0.0, std=1.0),
    ])
    estimator = make_pipeline(
        RegressionModel(net, transforms=preproc))
    return estimator


if __name__ == '__main__':
    estimator = get_estimator(mock=True).fit(None)
    estimator.predict(np.random.random((32, 2122945)))
