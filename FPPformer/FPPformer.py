# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
from utils.RevIN import RevIN
from FPPformer.Modules import *
from FPPformer.embed import DataEmbedding


class Encoder_process(nn.Module):
    def __init__(self, patch_num, encoder_num, encoders):
        super(Encoder_process, self).__init__()
        self.patch_num = patch_num
        self.encoder_num = encoder_num
        self.encoders = encoders

    def forward(self, x_enc):
        B, V, L, D = x_enc.shape
        x_patch_attn = x_enc.contiguous().view(B, V, self.patch_num, -1, D)

        encoder_out_list = []
        for i in range(self.encoder_num):
            x_out, x_patch_attn = self.encoders[i](x_patch_attn)
            encoder_out_list.append(x_out)
        return x_patch_attn.contiguous().view(B, V, -1, D), encoder_out_list


class FPPformer(nn.Module):
    def __init__(self, input_len, pred_len, encoder_layer=3, patch_size=12, d_model=4, dropout=0.05):
        super(FPPformer, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.patch_num = input_len // patch_size
        self.patch_size = patch_size
        self.encoder_num = encoder_layer
        self.d_model = d_model

        self.revin = RevIN(0, affine=False)
        self.Embed1 = DataEmbedding(d_model)
        self.Embed2 = DataEmbedding(d_model, start_pos=input_len)

        self.encoders = [Encoder(self.patch_size * 2 ** i, d_model, dropout)
                         for i in range(encoder_layer)]
        self.encoders = nn.ModuleList(self.encoders)
        self.encoder_process = Encoder_process(self.patch_num, self.encoder_num, self.encoders)
        self.b_patch_size = self.patch_size * 2 ** (self.encoder_num - 1)

        self.decoders = [Decoder(self.patch_size * 2 ** (encoder_layer - 1 - i), d_model, dropout)
                         for i in range(encoder_layer)]
        self.decoders = nn.ModuleList(self.decoders)
        # tackle the problem when the prediction sequence length is not the multiple integer of
        # the patch size at certain stage
        self.total_len = math.ceil(self.pred_len / self.b_patch_size) * self.b_patch_size

        self.projection1_0 = nn.Linear(d_model, 1)
        self.projection1_1 = nn.Sequential(nn.Linear(input_len,
                                                     max(2 * input_len, 2 * pred_len)),
                                           nn.Linear(max(2 * input_len, 2 * pred_len),
                                                     pred_len),
                                           )
        self.projection2 = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, V = x.shape
        x_enc = self.revin(x, 'norm')
        x_enc = x_enc.unsqueeze(-1)
        x_enc = self.Embed1(x_enc).transpose(1, 2)

        x_patch_attn, encoder_out_list = self.encoder_process(x_enc)
        x_map1 = self.projection1_0(x_patch_attn).squeeze(-1)
        x_out1 = self.projection1_1(x_map1).transpose(1, 2)

        x_dec = torch.zeros(B, self.pred_len, V, 1).to(x_enc.device)
        x_dec = self.Embed2(x_dec, pos_only=True).transpose(1, 2). \
            expand(B, V, self.pred_len, self.d_model)  # [B V L_pred D]
        # Only when the prediction sequence length is not the multiple integer of the patch size at certain stage
        if self.total_len > self.pred_len:
            x_dec = torch.cat([x_enc[:, :, self.pred_len - self.total_len:, :], x_dec], dim=2)

        x_dec = x_dec.contiguous().view(B, V, -1, self.b_patch_size, self.d_model)

        for i in range(self.encoder_num):
            x_dec = self.decoders[i](x_dec, encoder_out_list[-1 - i])

        x_map2 = self.projection2(x_dec.contiguous().view(B, V, -1, self.d_model)).squeeze(-1)
        x_out2 = x_map2.transpose(1, 2)
        x_out = x_out1 + x_out2[:, -self.pred_len:, :]
        x_out = self.revin(x_out, 'denorm')
        return x_out
