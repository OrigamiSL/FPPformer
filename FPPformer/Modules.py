import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utils.masking import OffDiagMask_PointLevel, OffDiagMask_PatchLevel, TriangularCausalMask


class Attn_PointLevel(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Attn_PointLevel, self).__init__()
        self.query_projection = nn.Linear(d_model, d_model)
        self.kv_projection = nn.Linear(d_model, d_model)

        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask='Diag'):
        B, V, P, L, D = queries.shape
        _, _, _, S, D = keys.shape
        scale = 1. / math.sqrt(D)

        queries = self.query_projection(queries)
        keys = self.kv_projection(keys)
        values = self.kv_projection(values)

        scores = torch.einsum("bvpld,bvpmd->bvplm", queries, keys)  # [B V P L L]

        if mask == 'Diag':
            attn_mask = OffDiagMask_PointLevel(B, V, P, L, device=queries.device)  # [B V P L L]
            scores.masked_fill_(attn_mask.mask, -np.inf)
        elif mask == 'Causal':
            assert (L == S)
            attn_mask = TriangularCausalMask(B, V, P, L, device=queries.device)  # [B V P L L ]
            scores.masked_fill_(attn_mask.mask, -np.inf)
        else:
            pass

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B V P L L]
        out = torch.einsum("bvplm,bvpmd->bvpld", attn, values)  # [B V P L D]

        return self.out_projection(out)  # [B V P L D]


class Attn_PatchLevel(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Attn_PatchLevel, self).__init__()
        self.query_projection = nn.Linear(d_model, d_model)
        self.kv_projection = nn.Linear(d_model, d_model)

        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask='Diag'):
        B, V, P, D = queries.shape
        _, _, S, D = keys.shape
        scale = 1. / math.sqrt(D)

        queries = self.query_projection(queries)
        keys = self.kv_projection(keys)
        values = self.kv_projection(values)

        scores = torch.einsum("bvpd,bvsd->bvps", queries, keys)  # [B V P P]
        if mask == 'Diag':
            attn_mask = OffDiagMask_PatchLevel(B, V, P, device=queries.device)  # [B V P P]
            scores.masked_fill_(attn_mask.mask, -np.inf)
        else:
            pass

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B V P P]
        out = torch.einsum("bvps,bvsd->bvpd", attn, values)  # [B V P D]

        return self.out_projection(out)  # [B V P D]


class Attn_VarLevel(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Attn_VarLevel, self).__init__()
        self.query_projection = nn.Linear(d_model, d_model)
        self.kv_projection = nn.Linear(d_model, d_model)

        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        B, P, V, D = queries.shape
        _, _, R, _ = keys.shape
        scale = 1. / math.sqrt(D)

        queries = self.query_projection(queries)
        keys = self.kv_projection(keys)
        values = self.kv_projection(values)

        scores = torch.einsum("bpvd,bprd->bpvr", queries, keys)  # [B P V V]

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bpvr,bprd->bpvd", attn, values)  # [B P V D]

        return self.out_projection(out)  # [B P V D]


class Encoder(nn.Module):
    def __init__(self, patch_size, d_model, dropout=0.1):
        super(Encoder, self).__init__()
        self.patch_dim = patch_size * d_model
        self.attn1 = Attn_PointLevel(d_model, dropout)
        self.attn2 = Attn_PatchLevel(self.patch_dim, dropout)

        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(self.patch_dim)
        self.norm3 = nn.LayerNorm(self.patch_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.patch_dim, 4 * self.patch_dim)
        self.linear2 = nn.Linear(4 * self.patch_dim, self.patch_dim)

    def forward(self, x):
        B, V, P, L, D = x.shape
        attn1_x = self.attn1(x, x, x, mask='Diag')
        x = (x + self.dropout(attn1_x)).contiguous().view(B, V, P, -1)
        x = self.norm1(x)

        attn2_x = self.attn2(x, x, x, mask='Diag').contiguous()

        y = x = self.norm2(x + self.dropout(attn2_x))
        y = self.activation(self.linear1(y))
        x = x + self.dropout(self.linear2(y))

        x_out = self.norm3(x)
        x_next = x_out.contiguous().view(B, V, P // 2, 2, L, D) \
            .view(B, V, P // 2, 2 * L, D)

        return x_out.view(B, V, P, -1), x_next


class Encoder_Cross(nn.Module):
    def __init__(self, patch_size, d_model, dropout=0.1):
        super(Encoder_Cross, self).__init__()
        self.patch_dim = patch_size * d_model
        self.attn1 = Attn_PointLevel(d_model, dropout)
        self.attn2 = Attn_PatchLevel(self.patch_dim, dropout)
        self.attn3 = Attn_VarLevel(self.patch_dim, dropout)

        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(self.patch_dim)
        self.norm3 = nn.LayerNorm(self.patch_dim)

        self.norm4 = nn.LayerNorm(self.patch_dim)
        self.norm5 = nn.LayerNorm(self.patch_dim)

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.patch_dim, 4 * self.patch_dim)
        self.linear2 = nn.Linear(4 * self.patch_dim, self.patch_dim)
        self.linear3 = nn.Linear(self.patch_dim, 4 * self.patch_dim)
        self.linear4 = nn.Linear(4 * self.patch_dim, self.patch_dim)

    def forward(self, x):
        B, V, P, L, D = x.shape
        attn1_x = self.attn1(x, x, x, mask='Diag')
        x = (x + self.dropout(attn1_x)).contiguous().view(B, V, P, -1)
        x = self.norm1(x)

        attn2_x = self.attn2(x, x, x, mask='Diag').contiguous()

        y = x = self.norm2(x + self.dropout(attn2_x))
        y = self.activation(self.linear1(y))
        x = x + self.dropout(self.linear2(y))

        x = self.norm3(x).permute(0, 2, 1, 3)  # B, P, V, LD
        z = x = self.norm4(x + self.dropout(self.attn3(x, x, x)))
        z = self.activation(self.linear3(z))
        x = x + self.dropout(self.linear4(z))

        x_out = self.norm5(x).permute(0, 2, 1, 3)  # B, V, P, LD
        x_next = x_out.contiguous().view(B, V, P // 2, 2, L, D) \
            .view(B, V, P // 2, 2 * L, D)

        return x_out.view(B, V, P, -1), x_next


class Decoder(nn.Module):
    def __init__(self, patch_size, d_model, dropout=0.1):
        super(Decoder, self).__init__()
        self.patch_dim = patch_size * d_model
        self.attn1 = Attn_PatchLevel(self.patch_dim, dropout)
        self.attn2 = Attn_PointLevel(d_model, dropout)

        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(self.patch_dim)
        self.norm3 = nn.LayerNorm(self.patch_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.patch_dim, 4 * self.patch_dim)
        self.linear2 = nn.Linear(4 * self.patch_dim, self.patch_dim)

    def forward(self, x, y):
        B, V, P, L, D = x.shape
        x = x.contiguous().view(B, V, P, -1)
        attn1_x = self.attn1(x, y, y, mask=None)
        x = self.norm1(x + self.dropout(attn1_x))

        x = x.contiguous().view(B, V, P, L, D)
        attn2_x = self.attn2(x, x, x, mask=None).contiguous().view(B, V, P, -1)

        y = x = self.norm2(x.contiguous().view(B, V, P, -1) + self.dropout(attn2_x))
        y = self.activation(self.linear1(y))
        x = x + self.dropout(self.linear2(y))
        x = self.norm3(x).contiguous().view(B, V, P, 2, L // 2, D) \
            .view(B, V, 2 * P, L // 2, D)

        return x


class Decoder_Cross(nn.Module):
    def __init__(self, patch_size, d_model, dropout=0.1):
        super(Decoder_Cross, self).__init__()
        self.patch_dim = patch_size * d_model
        self.attn1 = Attn_PatchLevel(self.patch_dim, dropout)
        self.attn2 = Attn_PointLevel(d_model, dropout)
        self.attn3 = Attn_VarLevel(self.patch_dim, dropout)

        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(self.patch_dim)
        self.norm3 = nn.LayerNorm(self.patch_dim)
        self.norm4 = nn.LayerNorm(self.patch_dim)
        self.norm5 = nn.LayerNorm(self.patch_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.patch_dim, 4 * self.patch_dim)
        self.linear2 = nn.Linear(4 * self.patch_dim, self.patch_dim)
        self.linear3 = nn.Linear(self.patch_dim, 4 * self.patch_dim)
        self.linear4 = nn.Linear(4 * self.patch_dim, self.patch_dim)

    def forward(self, x, y):
        B, V, P, L, D = x.shape
        x = x.contiguous().view(B, V, P, -1)
        attn1_x = self.attn1(x, y, y, mask=None)
        x = self.norm1(x + self.dropout(attn1_x))

        x = x.contiguous().view(B, V, P, L, D)
        attn2_x = self.attn2(x, x, x, mask=None).contiguous().view(B, V, P, -1)

        y = x = self.norm2(x.contiguous().view(B, V, P, -1) + self.dropout(attn2_x))
        y = self.activation(self.linear1(y))
        x = x + self.dropout(self.linear2(y))

        x = self.norm3(x).permute(0, 2, 1, 3)  # B, P, V, LD
        z = x = self.norm4(x + self.dropout(self.attn3(x, x, x)))
        z = self.activation(self.linear3(z))
        x = x + self.dropout(self.linear4(z))

        x = self.norm5(x).contiguous().view(B, V, P, 2, L // 2, D) \
            .view(B, V, 2 * P, L // 2, D)

        return x
