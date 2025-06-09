import torch
import torch.nn as nn
import omegaconf
import torch.nn.functional as F
import diffusers 

from .unet_utils import number_of_features_per_level
from .unet_blocks import TimeAwareDoubleConv, create_decoders, create_encoders, TimestepEmbedder, TimestepEmbedding


class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        conv_upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        upsample (str): algorithm used for decoder upsampling:
            InterpolateUpsampling:   'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'
            TransposeConvUpsampling: 'deconv'
            No upsampling:           None
            Default: 'default' (chooses automatically)
        dropout_prob (float or tuple): dropout probability, default: 0.1
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, temb_dim, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=False, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, conv_upscale=2, upsample='default', dropout_prob=0.1, is3d=True):
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size,
                                        conv_padding, conv_upscale, dropout_prob,
                                        layer_order, num_groups, pool_kernel_size, is3d, temb_dim=temb_dim)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding,
                                        layer_order, num_groups, upsample, dropout_prob,
                                        is3d, temb_dim=temb_dim)

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        self.final_activation = None

    def forward(self, x, temb, return_logits=False):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W) for 3D or (N, C, H, W) for 2D,
                              where N is the batch size, C is the number of channels,
                              D is the depth, H is the height, and W is the width.
            return_logits (bool): If True, returns both the output and the logits.
                                  If False, returns only the output. Default is False.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
                          If return_logits is True, returns a tuple of (output, logits).
        """
        output, logits = self._forward_logits(x, temb)
        if return_logits:
            return output, logits
        return output

    def _forward_logits(self, x, temb):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x, temb)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x, temb)

        x = self.final_conv(x)
        return x, x


class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, temb_channels=512, final_sigmoid=False, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=False, conv_padding=1,
                 conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=TimeAwareDoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     conv_upscale=conv_upscale,
                                     upsample=upsample,
                                     dropout_prob=dropout_prob,
                                     is3d=True,
                                     temb_dim=temb_channels)

        self.time_proj = TimestepEmbedder(temb_channels)
        self.act = nn.SiLU()

    def forward(self, x, time_tens):
        '''
        x: torch.Tensor. Voxel of shape (B, X,Y,Z)
        time_tens: Shape (B, )

        Returns: logits of shape (B, X, Y, Z, vocab_len)
        '''
        x = x.float()
        # add channel 1 if missing
        if len(x.shape) < 5:
          x = x.unsqueeze(dim=1)

        # permute to (N, C, D, H, W)
        x = torch.permute(x, dims=(0, 1, 4, 3, 2))

        # get sinusoidal embeddings
        temb = self.time_proj(time_tens)

        logits = super().forward(x, temb, False)

        # change order to (N, X, Y, Z, C)
        final_output = torch.permute(logits, dims=(0, 4, 3, 2, 1))  
        return final_output
    






'''
class DIT(nn.Module):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size

    self.vocab_embed = EmbeddingLayer(config.model.hidden_size,
                                      vocab_size)
    self.sigma_map = TimestepEmbedder(config.model.cond_dim)
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads)

    blocks = []
    for _ in range(config.model.n_blocks):
      blocks.append(DDiTBlock(config.model.hidden_size,
                              config.model.n_heads,
                              config.model.cond_dim,
                              dropout=config.model.dropout))
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      vocab_size,
      config.model.cond_dim)
    self.scale_by_sigma = config.model.scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, voxel, sigma):
    #x = self.vocab_embed(indices)
    c = F.silu(self.sigma_map(sigma))

    rotary_cos_sin = self.rotary_emb(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
      x = self.output_layer(x, c)

    return x
'''