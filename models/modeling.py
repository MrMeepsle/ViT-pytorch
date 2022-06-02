# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

import models.configs as configs
from .modeling_resnet import ResNetV2

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis, img_size):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            self.n_patches = (img_size[0] // 16) * (img_size[1] // 16)
        else:
            patch_size = _pair(config.patches["size"])
            self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.relative_pos_embedding = nn.Parameter(torch.zeros(1, self.n_patches + 1, config.hidden_size),
                                                   requires_grad=False)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        B = mixed_query_layer.shape[0]
        relative_pos_embedding = self.relative_pos_embedding.expand(B, -1, -1)
        mixed_key_layer = self.key(hidden_states) + relative_pos_embedding
        mixed_value_layer = self.value(hidden_states) + relative_pos_embedding

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size), requires_grad=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis, img_size):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis, img_size)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class PosEncoding(nn.Module):
    def __init__(self, size, type, plot=True):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PosEncoding, self).__init__()
        self.type = type
        self.plot = plot
        self.size = size  ### Size of input dimension

        self.cached_embedding = None

        if (self.type == "sin_cos"):
            """
            Adapted from https://github.com/tatp22/multidim-positional-encoding
            License: https://github.com/tatp22/multidim-positional-encoding/blob/master/LICENSE
            """
            output_channels = int(np.ceil(self.size[-1] / 2) * 2)
            self.channels = output_channels
            inv_freq = 1.0 / (10000 ** (torch.arange(0, output_channels, 2).float() / output_channels))
            self.register_buffer("inv_freq", inv_freq)
        elif (self.type == "arctan"):
            """
            Adapted from https://github.com/tatp22/multidim-positional-encoding
            License: https://github.com/tatp22/multidim-positional-encoding/blob/master/LICENSE
            """
            output_channels = int(np.ceil(self.size[-1] / 2) * 2)
            self.channels = output_channels
            self.alpha = 10

        elif (self.type == "RPEsin"):
            """
            Adapted from https://github.com/tatp22/multidim-positional-encoding
            License: https://github.com/tatp22/multidim-positional-encoding/blob/master/LICENSE
            """
            output_channels = int(np.ceil(self.size[-1] / 2) * 2)
            self.channels = output_channels
            self.alpha = 10

        elif (self.type == "linear"):
            output_channels = int(np.ceil(self.size[-1] / 2) * 2)
            self.channels = output_channels
            self.alpha = 4
            self.beta = 0.2

    def forward(self):
        if (self.type == "zeros"):
            self.cached_embedding = torch.zeros(self.size)

        elif (self.type == "random"):
            self.cached_embedding = torch.rand(self.size)

        elif (self.type == "sin_cos"):
            """
            :param tensor: A 3d tensor of size (batch_size, x, ch)
            :return: Positional Encoding Matrix of size (batch_size, x, ch)
            """
            if self.cached_embedding is not None:
                return self.cached_embedding

            self.cached_embedding = None
            batch_size, x, orig_ch = self.size
            pos_x = torch.arange(x).type(self.inv_freq.type())
            sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
            emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
            embedding = torch.zeros((x, self.channels))
            embedding[:, : self.channels] = emb_x

            self.cached_embedding = embedding[None, :, :orig_ch].repeat(batch_size, 1, 1)

        elif (self.type == "arctan"):
            """
            :param tensor: A 3d tensor of size (batch_size, x, ch)
            :return: Positional Encoding Matrix of size (batch_size, x, ch)
            """
            if self.cached_embedding is not None:
                return self.cached_embedding

            self.cached_embedding = None
            batch_size, x, orig_ch = self.size  ## x = amount of inputs, orig_ch = input dimension
            embedding = torch.zeros((x, self.channels))
            for i in range(x):
                for j in range(orig_ch):
                    embedding[i, j] = np.pi / 2 - abs(np.arctan((i) - (j) / self.alpha))

            # embedding[:, : self.channels] = emb_x

            self.cached_embedding = embedding[None, :, :self.channels].repeat(batch_size, 1, 1)

        elif (self.type == "RPEsin"):
            """
            :param tensor: A 3d tensor of size (batch_size, x, ch)
            :return: Positional Encoding Matrix of size (batch_size, x, ch)
            """
            if self.cached_embedding is not None:
                return self.cached_embedding

            self.cached_embedding = None
            batch_size, x, orig_ch = self.size ## x = amount of inputs, orig_ch = input dimension
            embedding = torch.zeros((x, self.channels))
            for i in range(x):
                for j in range(orig_ch):
                    embedding[i,j] = np.sin((j-i)/(10000**(2*i/orig_ch)))

            # embedding[:, : self.channels] = emb_x

            self.cached_embedding = embedding[None, :, :self.channels].repeat(batch_size, 1, 1)

        elif (self.type == "linear"):
            """
            :param tensor: A 3d tensor of size (batch_size, x, ch)
            :return: Positional Encoding Matrix of size (batch_size, x, ch)
            """
            if self.cached_embedding is not None:
                return self.cached_embedding

            self.cached_embedding = None
            batch_size, x, orig_ch = self.size  ## x = amount of inputs, orig_ch = input dimension
            embedding = torch.zeros((x, self.channels))
            for i in range(x):
                for j in range(orig_ch):
                    embedding[i, j] = self.alpha * i + self.beta * j

            embedding = embedding * (1 / (torch.max(embedding) + 10 ** -3))

            embedding = embedding * (1/(torch.max(embedding)+10**-3))

            self.cached_embedding = embedding[None, :, :orig_ch].repeat(batch_size, 1, 1)

        # if self.plot:
        #     plt.imshow(self.cached_embedding.numpy()[0, :, :], cmap='hot', interpolation='nearest')
        #     plt.show()
        #     plt.clf()

        return self.cached_embedding


class Encoder(nn.Module):
    def __init__(self, config, vis, img_size):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis, img_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, img_size)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

    def init_from_scratch(self, pos_encoding, encoding_type):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)

            self.transformer.embeddings.cls_token.copy_(torch.rand(self.transformer.embeddings.cls_token.size()))

            if (encoding_type == "absolute"):
                encoding = PosEncoding(self.transformer.embeddings.position_embeddings.size(), pos_encoding)
                self.transformer.embeddings.position_embeddings.copy_(encoding())
            elif (encoding_type == "relative"):

                for bname, block in self.transformer.encoder.named_children():
                    for uname, unit in block.named_children():
                        encoding = PosEncoding(unit.attn.relative_pos_embedding.size(), pos_encoding)
                        unit.attn.relative_pos_embedding.copy_(encoding())

    def load_from(self, weights):
        # this loads in the pretrained model
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_7': configs.get_b7_config(),
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
