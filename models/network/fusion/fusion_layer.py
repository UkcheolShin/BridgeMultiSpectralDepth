import torch
import torch.nn as nn
from models.network.fusion.swin_transformer import BasicSwinLayer


class ChannelMerging(nn.Module):
    """ Channel Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, indim, outdim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(indim, outdim, bias=False)
        self.norm = norm_layer(indim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class FuseModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.dim = dim*3
        self.indim = dim//2*4
        self.outdim = dim
        self.fuse = BasicSwinLayer(dim=self.indim, depth=1, num_heads=4, mode='self', window_size=7)
        self.merge = ChannelMerging(indim=self.indim, outdim=self.outdim)

    def forward(self, x):
        B_, C_, H_, W_ = x.shape
        # flatten
        x = x.flatten(2).transpose(1,2)

        x = self.fuse(x, H_, W_)[0]
        x = self.merge(x, H_, W_)

        # deflatten
        x = x.view(B_,H_,W_,-1).permute(0, 3, 1, 2).contiguous()

        return x

class FeatFuseModule(nn.Module):
    def __init__(self, inplanes, active='L12345') :
        super(FeatFuseModule, self).__init__()
        self.inplanes = inplanes
        self.active = active
        if '1' in self.active:
            self.fuse_layer1 = FuseModule(dim=self.inplanes[0])
        if '2' in self.active:
            self.fuse_layer2 = FuseModule(dim=self.inplanes[1])
        if '3' in self.active:
            self.fuse_layer3 = FuseModule(dim=self.inplanes[2])
        if '4' in self.active:
            self.fuse_layer4 = FuseModule(dim=self.inplanes[3])
        if '5' in self.active:
            self.fuse_layer5 = FuseModule(dim=self.inplanes[4])

    def forward(self, features) :
        if '1' in self.active:
            features[0] = self.fuse_layer1(features[0])
        if '2' in self.active:
            features[1] = self.fuse_layer2(features[1])
        if '3' in self.active:
            features[2] = self.fuse_layer3(features[2])
        if '4' in self.active:
            features[3] = self.fuse_layer4(features[3])
        if '5' in self.active:
            features[4] = self.fuse_layer5(features[4])
        return features

