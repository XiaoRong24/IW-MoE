import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils.tf_spatial_transform_local as tf_spatial_transform_local
import utils.torch_tps_transform as torch_tps_transform
import utils.tf_mesh2flow as tf_mesh2flow
import utils.constant as constant
import utils.torch_tps2flow as torch_tps2flow

from timm.layers.helpers import to_2tuple
from timm.models.layers import DropPath, trunc_normal_
from net.mambablock import MambaBlock, Conv2d_BN, RepDW

grid_w = constant.GRID_W
grid_h = constant.GRID_H
gpu_device = constant.GPU_DEVICE

def shift2mesh0(mesh_shift, height,width):
    device = mesh_shift.device
    batch_size = mesh_shift.shape[0]
    h = height / grid_h
    w = width / grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            p = torch.FloatTensor([ww, hh])
            ori_pt.append(p.unsqueeze(0))
    ori_pt = torch.cat(ori_pt,dim=0)
    # print(ori_pt.shape)
    ori_pt = ori_pt.view(grid_h + 1, grid_w + 1, 2)
    # print(ori_pt)
    ori_pt = torch.tile(ori_pt.unsqueeze(0), [batch_size, 1, 1, 1])
    ori_pt = ori_pt.to(gpu_device)
    # print("ori_pt:",ori_pt.shape)
    # print("mesh_shift:", mesh_shift.shape)
    tar_pt = ori_pt + mesh_shift
    return tar_pt

def get_rigid_mesh(batch_size, height, width):

    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    ww = ww.to(gpu_device)
    hh = hh.to(gpu_device)

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2
    # norm_mesh = torch.stack([mesh_h, mesh_w], 3)  # bs*(grid_h+1)*(grid_w+1)*2
    # print("norm_mesh:",norm_mesh.shape)
    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2

def H2Mesh(H, rigid_mesh):

    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    ori_pt = ori_pt.to(gpu_device)
    ones = ones.to(gpu_device)

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1)) # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])

    return mesh


def autopad2(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad2(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=nn.SiLU(), deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return self.cv2(self.cv1(x))



class ConvBlock(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))




def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # For rare cases, the attention weights are inf due to the mix-precision training.
        # We clamp the tensor to the max values of the current data type
        # This is different from MAE training as we don't observe such cases on image-only MAE.
        if torch.isinf(attn).any():
            clamp_value = torch.finfo(attn.dtype).max - 1000
            attn = torch.clamp(attn, min=-clamp_value, max=clamp_value)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 task_num=9,
                 noisy_gating=True,
                 att_w_topk_loss=0.0, att_limit_k=0,
                 cvloss=0, switchloss=0.01 * 1, zloss=0.001 * 1, w_topk_loss=0.0, limit_k=0,
                 w_MI=0., num_attn_experts=24, sample_topk=0, moe_type='normal'
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)


        self.mixer = Attention(
        dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_norm=qk_scale,
        attn_drop=attn_drop,
        proj_drop=drop,
        norm_layer=norm_layer,
        )


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x



class SelfAttentionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 transformer_blocks=1,
    ):


        super().__init__()

        self.blocks = nn.ModuleList([Block(dim=dim,
                                           num_heads=num_heads,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias,
                                           qk_scale=qk_scale,
                                           drop=drop,
                                           attn_drop=attn_drop,
                                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                           layer_scale=layer_scale)
                                     for i in range(transformer_blocks)])

        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape


        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:

            x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))

            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W

        x = window_partition(x, self.window_size)

        for _, blk in enumerate(self.blocks):
            x = blk(x)

        x = window_reverse(x, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()

        return x

class Downsample(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self,in_chans=3, embed_dim=48, patch_size=16, stride=2, padding=0):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = Conv2d_BN(in_chans, embed_dim, patch_size, stride, padding)

    def forward(self, x):
        x = self.proj(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self,inchannels=3,mlp_ratio=4):
        super(FeatureExtractor, self).__init__()
        self.mlp_ratio = mlp_ratio
        # Conv
        self.featureExtractor = nn.Sequential(
            Downsample(inchannels, 64, 3, 2, 1),  # 512 -> 256
            Downsample(64, 64, 3, 2, 1),  # 256 -> 128
            ConvBlock(64, 64),
            Downsample(64, 64, 3, 2, 1),  # 128 -> 64
            ConvBlock(64,64),
            Downsample(64, 64, 3, 2, 1),  # 64 -> 32
            ConvBlock(64, 64),
            SelfAttentionLayer(dim=64, num_heads=2, window_size=8, mlp_ratio=mlp_ratio, qkv_bias=True,
                               drop=0, attn_drop=0, drop_path=0, layer_scale=None),
            Downsample(64, 64, 3, 2, 1),  # 32 -> 16
            ConvBlock(64, 64),
            Downsample(64, 64, 3, 2, 1),  # 16 -> 8
            ConvBlock(64, 64),
            Downsample(64, 64, 3, 2, 1),  # 8 -> 4
        )

    def forward(self,x):
        x = self.featureExtractor(x)
        return x



class RegressionNetwork(nn.Module):
    def __init__(self):
        super(RegressionNetwork, self).__init__()
        self.patch_height = (grid_h + 1)
        self.patch_width = (grid_w + 1)


        self.head = nn.Sequential(
            nn.Conv2d(64,1024,(3,4),2),
            nn.Flatten(),
            # nn.Linear(768, 1024),
            nn.SiLU(inplace=True),
            #nn.GELU(),
            nn.Linear(1024, self.patch_height * self.patch_width * 2)
        )

    def forward(self,x):
        x = self.head(x)

        return x.view(-1,self.patch_height, self.patch_width, 2)




class BackboneNetwork(nn.Module):
    def __init__(self):
        super(BackboneNetwork, self).__init__()

        self.feature_extractor = FeatureExtractor(3)
        self.regression = RegressionNetwork()

    def forward(self,x):
        x = self.feature_extractor(x)
        out = self.regression(x)

        return out


class DSConvBlock(nn.Module):
    """
    Depthwise Conv -> Pointwise Conv -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(DSConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(self.pointwise(self.depthwise(x)))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # 统一和减少通道数
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU6(),
            RepDW(32)  # 使用 RepDW 模块
        )

        self.conv2 = nn.Sequential(DSConvBlock(32, 64), RepDW(64))
        self.conv3 = nn.Sequential(DSConvBlock(64, 128), RepDW(128))
        self.conv4 = nn.Sequential(DSConvBlock(128, 128), RepDW(128))
        self.conv5 = nn.Sequential(DSConvBlock(128, 256), RepDW(256))

    def forward(self, x):
        feature = []

        # Conv Block 1
        x1 = self.conv1(x)
        feature.append(x1)
        x = F.max_pool2d(x1, 2, 2)

        # Conv Block 2
        x2 = self.conv2(x)
        feature.append(x2)
        x = F.max_pool2d(x2, 2, 2)

        # Conv Block 3
        x3 = self.conv3(x)
        feature.append(x3)
        x = F.max_pool2d(x3, 2, 2)

        # Conv Block 4
        x4 = self.conv4(x)
        feature.append(x4)
        x = F.max_pool2d(x4, 2, 2)

        # Conv Block 5
        x5 = self.conv5(x)
        feature.append(x5)

        return feature


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # 统一和减少通道数
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv_block1 = nn.Sequential(DSConvBlock(256, 128), RepDW(128))

        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.conv_block2 = nn.Sequential(DSConvBlock(256, 128), RepDW(128))

        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv_block3 = nn.Sequential(DSConvBlock(128, 64), RepDW(64))

        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv_block4 = nn.Sequential(DSConvBlock(64, 32), RepDW(32))

        self.flow = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, padding=0)

    def forward(self, feature):
        # Apply the first block of deconvolution and convolutions
        h_deconv1 = self.deconv1(feature[-1])
        h_deconv_concat1 = torch.cat([feature[-2], h_deconv1], dim=1)
        conv1 = self.conv_block1(h_deconv_concat1)

        # Apply the second block
        h_deconv2 = self.deconv2(conv1)
        h_deconv_concat2 = torch.cat([feature[-3], h_deconv2], dim=1)
        conv2 = self.conv_block2(h_deconv_concat2)

        # Apply the third block
        h_deconv3 = self.deconv3(conv2)
        h_deconv_concat3 = torch.cat([feature[-4], h_deconv3], dim=1)
        conv3 = self.conv_block3(h_deconv_concat3)

        # Apply the fourth block
        h_deconv4 = self.deconv4(conv3)
        h_deconv_concat4 = torch.cat([feature[-5], h_deconv4], dim=1)
        conv4 = self.conv_block4(h_deconv_concat4)

        # Generate flow
        flow = self.flow(conv4)

        return flow


class TaskClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # 1. Point coordinate processing branch
        self.point_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((6, 8))
        )

        # 2. New feature processing branch (processing 4D image features)
        self.feat_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [8,32,192,256]
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),  # [8,32,96,128]

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [8,64,48,64]
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((3, 4)),  # [8,64,3,4]
            nn.Flatten(1)  # [8,64 * 3 * 4]
        )

        # 3. Fusion classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 8 + 64 * 3 * 4, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, points, img_feat):
        """
        points: [B, H, W, 2] -> [8,13,17,2]
        img_feat: [B, C, H, W] -> [8,3,384,512]
        """
        # Point coordinate processing
        points = points.permute(0, 3, 1, 2)  # [8,2,13,17]
        point_feat = self.point_encoder(points)  # [8,128,6,8]
        point_feat = point_feat.flatten(1)  # [8,128 * 6 * 8]

        # Image feature processing
        img_feat = self.feat_encoder(img_feat)  # [8,64 * 3 * 4]

        # feature fusion
        combined = torch.cat([point_feat, img_feat], dim=1)  # [8, 128 * 6 * 8 + 64 * 3 * 4]

        return self.classifier(combined)




class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [8, 3, 256, 256] -> [8, 3, 1, 1]
            nn.Flatten(),  # [8, 3]
            nn.Linear(3, expert_number),
        )
        self.expert_number = expert_number
        self.top_k = top_k

    def forward(self, hidden_states):
        # logits
        router_logits = self.gate(hidden_states)  # shape is (b , expert_number)


        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)

        # Calculate the output of top k experts
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )  # # [B, top_k]

        # The normalization of expert rights
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        # router_weights = router_weights.to(hidden_states.dtype)

        # Generate expert mask
        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.expert_number
        )  # [B, top_k, expert_number]
        expert_mask = expert_mask.permute(2, 1, 0)  # [expert_number, top_k, B]

        return router_logits, router_weights, selected_experts, expert_mask

class TaskAwareMOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k, num_classes):
        super().__init__()
        # Gated network input is a fusion of image features and task classification prediction features
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  #  [B, C, H, W] -> [B, C, 1, 1]
            nn.Flatten(),           # [B, C]，C = hidden_dim（or 3）
            nn.Linear(hidden_dim + num_classes, expert_number)  # The dimension after fusion
        )
        self.expert_number = expert_number
        self.top_k = top_k

    def forward(self, hidden_states, task_cls):
        # print(task_cls.shape)
        # Obtain the features of AdaptiveAvgPool2d and Flatten
        image_features = self.gate[:-1](hidden_states)

        # Use F.softmax to process task_cls into a probability distribution
        fused_features = torch.cat([image_features, F.softmax(task_cls, dim=-1)], dim=1)

        # The calculation of routing logits is now based on fused features
        router_logits = self.gate[-1](fused_features) # shape is (b, expert_number)

        # Calculate the probability of experts passing through softmax
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)

        # Calculate the output of top k experts
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )  # [B, top_k]

        # The normalization of expert rights
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)

        # Generate expert mask
        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.expert_number
        )  # [B, top_k, expert_number]
        expert_mask = expert_mask.permute(2, 1, 0)  # [expert_number, top_k, B]

        return router_logits, router_weights, selected_experts , expert_mask

class MOEConfig:
    def __init__(
            self,
            task_num,
            hidden_dim,
            expert_number,
            top_k,
            shared_experts_number,
            residual_experts_number,
            residual_top_k,
    ):
        self.task_num = task_num
        self.hidden_dim = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number
        self.residual_experts_number = residual_experts_number
        self.residual_top_k = residual_top_k



class SparseMOE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.expert_number = config.expert_number
        self.top_k = config.top_k

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    BackboneNetwork()
                )for _ in range(self.expert_number)
            ]
        )

        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.size()

        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(x)


        final_output = torch.zeros(
            (B, 13, 17, 2),
            dtype=x.dtype,
            device=x.device
        )  # [B, 64, 2, 2]

        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx] shape is (top_k, b * s)
            idx, top_x = torch.where(expert_mask[expert_idx])   # idx: [M], top_x: [M]

            if len(top_x) > 0:
                # Extract the batch data that the current expert needs to process
                current_input = x[top_x]  # [M, C, H, W]
                # Weighted output
                expert_output = expert_layer(current_input)  # [M, C, H, W]
                weighted_output = expert_output * router_weights[top_x, idx].view(-1, 1, 1, 1)
                # Accumulate to the final output
                # final_output[top_x] += weighted_output
                final_output.index_add_(0, top_x, weighted_output)

        return final_output, router_logits  # shape is (b * s, expert_number)

class DynamicResidualMOE(nn.Module):
    def __init__(self, config):

        super().__init__()
        self.expert_number = config.residual_experts_number
        self.top_k = config.residual_top_k


        self.experts = nn.ModuleList([
            nn.Sequential(Encoder(), Decoder())
            for _ in range(self.expert_number)
        ])

        self.router = TaskAwareMOERouter(config.hidden_dim, self.expert_number, self.top_k, 2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x,  active_residual_tasks, task_cls):

        B, C, H, W = x.size()

        task_pred = torch.argmax(task_cls, dim=-1)
        # print("task_pred", task_pred)

        # 1. Create a difficult task mask
        difficult_task_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        for task_id in active_residual_tasks:
            difficult_task_mask |= (task_pred == task_id)

        if not difficult_task_mask.any():
            return torch.zeros((B, 2, H, W), device=x.device, dtype=x.dtype), None

        # 2. Extract difficult task samples
        difficult_samples = x[difficult_task_mask]
        difficult_task_cls = task_cls[difficult_task_mask]

        active_task_indices = torch.tensor(sorted(active_residual_tasks), device=x.device)
        selected_task_cls = difficult_task_cls[:, active_task_indices]

        # print("difficult_samples", difficult_samples.shape)
        # print("selected_task_cls", selected_task_cls.shape)

        # 3. Sample of difficult routing tasks
        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(difficult_samples, selected_task_cls)

        residual_flow_difficult_samples = torch.zeros(
            (difficult_samples.size(0), 2, H, W),
            dtype=x.dtype,
            device=x.device
        )

        # 4. Traverse experts and calculate weighted outputs
        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]


            batch_indices_in_difficult_samples = (selected_experts_indices == expert_idx).nonzero(as_tuple=True)[0]

            if len(batch_indices_in_difficult_samples) > 0:
                current_input = difficult_samples[batch_indices_in_difficult_samples]

                # Obtain the corresponding routing weights
                weights_for_expert = router_weights[batch_indices_in_difficult_samples,
                (selected_experts_indices[batch_indices_in_difficult_samples] == expert_idx).nonzero(as_tuple=True)[
                    1]].view(-1, 1, 1, 1)

                expert_output = expert_layer(current_input)
                weighted_output = expert_output * weights_for_expert

                residual_flow_difficult_samples.index_add_(0, batch_indices_in_difficult_samples, weighted_output)

        # 5. Merge the residual streams back into the original batch structure
        final_residual_flow = torch.zeros((B, 2, H, W), device=x.device, dtype=x.dtype)
        final_residual_flow[difficult_task_mask] = residual_flow_difficult_samples

        return final_residual_flow, router_logits

class Slave_MOE(nn.Module):
    def __init__(self, config):
        super(Slave_MOE, self).__init__()

        self.task_classifier = TaskClassifier(config.task_num)

        self.DR_MOE = DynamicResidualMOE(config)

    def forward(self,mesh_motion, delta_flow,warp_image, fusion_warp_image, x, input_img, active_residual_tasks,  epoch):
        batch_size, _, height, width = x.shape

        # Task classifier
        task_cls = self.task_classifier(mesh_motion, x)

        if epoch >= 10:
            residual_flow, resdiue_router_logits = self.DR_MOE(fusion_warp_image,
                                                                          active_residual_tasks, task_cls)

            # Synthesize the final flow field
            warp_flow1 = warp_with_flow(delta_flow, delta_flow)
            flow1 = delta_flow + warp_flow1
            warp_flow2 = warp_with_flow(flow1, residual_flow)
            flow = residual_flow + warp_flow2
            final_image = warp_with_flow(input_img, flow)

        else:
            # The final flow field is delta_flow
            flow = delta_flow
            final_image = warp_image
            resdiue_router_logits = None
            residual_flow = torch.zeros_like(delta_flow)

        return task_cls, residual_flow, resdiue_router_logits, flow, final_image

class Master_MOE(nn.Module):
    def __init__(self, config):
        super(Master_MOE, self).__init__()

        self.sparse_moe = SparseMOE(config)
        self.shared_experts = nn.ModuleList(
            [nn.Sequential(BackboneNetwork())
             for _ in range(config.shared_experts_number)]
        )

        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape

        sparse_moe_out, router_logits = self.sparse_moe(x)

        shared_experts_out = [expert(x) for expert in self.shared_experts]
        shared_moe_out = torch.stack(shared_experts_out, dim=0).sum(dim=0)

        # Learning Fusion Weight
        gate_weight = self.fusion_gate(x).unsqueeze(-1).unsqueeze(-1)

        # Weighted fusion
        mesh_motion = gate_weight * sparse_moe_out + (1 - gate_weight) * shared_moe_out

        return mesh_motion, router_logits


class Master_Slave_MOE(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 用于存储中间结果的缓冲区
        self.moe_output_buffer = None
        self.shared_output_buffer = None
        self.fusion_weight_buffer = None

        self.master_moe = Master_MOE(config)
        self.slave_moe = Slave_MOE(config)


    def forward(self, x,input_img,mask_img, active_residual_tasks, epoch):
        batch_size, _, height, width = x.shape

        '''Master MoE architecture'''
        mesh_motion, router_logits = self.master_moe(x)

        rigid_mesh = get_rigid_mesh(batch_size, height, width)
        H_one = torch.eye(3)
        H = torch.tile(H_one.unsqueeze(0), [batch_size, 1, 1]).to(gpu_device)
        ini_mesh = H2Mesh(H, rigid_mesh)
        mesh_final = ini_mesh + mesh_motion
        norm_rigid_mesh = get_norm_mesh(rigid_mesh, height, width)
        norm_mesh = get_norm_mesh(mesh_final, height, width)

        delta_flow = torch_tps2flow.transformer(input_img, norm_rigid_mesh, norm_mesh, (height, width))
        output_tps = warp_with_flow(torch.cat((input_img, mask_img), 1), delta_flow)
        warp_image = output_tps[:, 0:3, ...]
        warp_mask = output_tps[:, 3:6, ...]
        fusion_warp_image = torch.mul(warp_image, warp_mask)

        '''Slave MoE architecture'''
        task_cls, residual_flow, resdiue_router_logits, flow, final_image = self.slave_moe(mesh_motion, delta_flow, warp_image, fusion_warp_image, x, input_img, active_residual_tasks,  epoch)


        '''Save intermediate results for monitoring purposes'''
        self.moe_output_buffer = delta_flow.detach()
        self.shared_output_buffer = mesh_motion.detach()

        return flow, final_image, router_logits, resdiue_router_logits, task_cls

class IWMoeNetwork(nn.Module):
    def __init__(self, num_experts=5, num_tasks=5):
        super(IWMoeNetwork, self).__init__()

        '''Hard parameter shared encoder'''
        self.SharedFeatureEncoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),

            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),
        )

        self.config = MOEConfig(6, 3, 12, 6, 6, 3,1)

        '''Dynamic parameter sharing expert - master-slave MoE architecture'''
        self.master_slave_moe = Master_Slave_MOE(self.config)

    # using tps strategy to warp the input image
    def forward(self, input_img, mask_img,active_residual_tasks, epoch):
        batch_size, _, height, width = input_img.shape

        shared_features = self.SharedFeatureEncoder(input_img)

        feature = torch.mul(shared_features, mask_img)

        delta_flow, warp_image, router_logits,resdiue_router_logits, task_cls = self.master_slave_moe(feature, input_img, mask_img,active_residual_tasks, epoch)

        return delta_flow, warp_image, router_logits, resdiue_router_logits, task_cls



def tensor_DLT(src_p, dst_p):
    bs, _, _ = src_p.shape

    ones = torch.ones(bs, 4, 1)
    if torch.cuda.is_available():
        ones = ones.to(gpu_device)
    xy1 = torch.cat((src_p, ones), 2)
    zeros = torch.zeros_like(xy1)
    if torch.cuda.is_available():
        zeros = zeros.to(gpu_device)

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
    M1 = torch.cat((xyu, xyd), 2).reshape(bs, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1),
        src_p.reshape(-1, 1, 2),
    ).reshape(bs, -1, 2)

    # Ah = b
    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(bs, -1, 1)

    # h = A^{-1}b
    Ainv = torch.inverse(A)
    h8 = torch.matmul(Ainv, b).reshape(bs, 8)

    H = torch.cat((h8, ones[:, 0, :]), 1).reshape(bs, 3, 3)
    return H



def warp_with_flow(img, flow):
    #initilize grid_coord
    batch, C, H, W = img.shape
    coords0 = torch.meshgrid(torch.arange(H).cuda(), torch.arange(W).cuda())
    coords0 = torch.stack(coords0[::-1], dim=0).float()
    coords0 = coords0[None].repeat(batch, 1, 1, 1)  # bs, 2, h, w

    # target coordinates
    target_coord = coords0 + flow

    # normalization
    target_coord_w = target_coord[:,0,:,:]*2./float(W) - 1.
    target_coord_h = target_coord[:,1,:,:]*2./float(H) - 1.
    target_coord_wh = torch.stack([target_coord_w, target_coord_h], 1)

    # warp
    warped_img = F.grid_sample(img, target_coord_wh.permute(0,2,3,1), align_corners=True)

    return warped_img
