from collections import OrderedDict
import math
from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .utils import to_2tuple


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


#here
class BiDirectionalCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(BiDirectionalCrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads)

    def forward(self, text_emb, image_emb):
        # text query image
        text_query_image = self.mha(text_emb, image_emb, image_emb)[0]
        # img query text
        image_query_text = self.mha(image_emb, text_emb, text_emb)[0]

        return text_query_image, image_query_text


# class mmadapter(nn.Module):
#     def __init__(self, img_size,text_size):
#         super(mmadapter, self).__init__()
#         self.img_proj_down = nn.Linear(img_size, 64)
#         self.img_proj_up = nn.Linear(64, img_size)
#         self.img_ln = nn.LayerNorm(img_size)
#         self.txt_proj_down = nn.Linear(text_size, 64)
#         self.txt_proj_up = nn.Linear(64, text_size)
#         self.txt_ln = nn.LayerNorm(text_size)
#         self.Biatten = BiDirectionalCrossAttention(64, 8)

#     def forward(self, img,text):
#         img_init = img
#         text_init = text
#         img = self.img_proj_down(img)
#         img = F.gelu(img)

#         text = self.txt_proj_down(text)
#         text = F.gelu(text)
    

#         text_query_image, image_query_text = self.Biatten(text, img)

#         img= img + (text_query_image + image_query_text)/2.0
#         text = text + (text_query_image + image_query_text)/2.0

#         img = self.img_proj_up(img)

#         img = img_init + self.img_ln(img)

#         text = self.txt_proj_up(text)

#         text = text_init + self.txt_ln(text)


#         return img, text
    
    
class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            layer_id:int = 0,
            mmadapter:Optional[nn.ModuleList] = None,
            mmadapter_aux:Optional[nn.ModuleList] = None
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value else nn.Identity()
        if mmadapter is not None:
            self.mmadapter = mmadapter[layer_id]
            self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        else:
            self.mmadapter = None
        # self.layer_id = layer_id
        # gate

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        xls1 = self.ls_1(self.attention(self.ln_1(x), attn_mask=attn_mask))
        x = x + xls1
        if self.mmadapter is not None:
            alpha = torch.sigmoid(self.gate1)
        xmlp = self.mlp(self.ln_2(x))
        if self.mmadapter is not None:
            xmlp =  alpha *self.mmadapter(xmlp) + (1 - alpha) * xmlp
        x = x + self.ls_2(xmlp)
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
           d_model, n_head,
           scaled_cosine=scale_cosine_attn,
           scale_heads=scale_heads,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            mmadapter:nn.Module = None,
            mmadapter_aux:nn.Module = None,
            align_before:bool = False
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.align_before = align_before

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer,layer_id=layer_id,mmadapter=mmadapter,mmadapter_aux=mmadapter_aux)
            for layer_id in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        intermediates = []  # save mid-layer features 
        num_resblocks = len(self.resblocks)
        for i, r in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
                if self.align_before:
                    if x is not None and i >= 5 and i < num_resblocks - 1:  # from 6th to 11th layer
                        intermediates.append(x)
        if self.align_before:            
            return x, intermediates
                
        return x
        # for r in self.resblocks:
        #     if self.grad_checkpointing and not torch.jit.is_scripting():
        #         x = checkpoint(r, x, attn_mask)
        #     else:
        #         x = r(x, attn_mask=attn_mask)
        # return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            output_dim: int = 512,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            mmadapter:nn.ModuleList = None,
            mmadapter_aux:nn.ModuleList = None,
            modalemb:nn.Module = None,
            align_before:bool = False,
    ):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)

        self.align_before = align_before

        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.ln_pre = norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mmadapter=mmadapter,
            mmadapter_aux=mmadapter_aux,
            align_before=align_before,
        )

        self.ln_post = norm_layer(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        # self.modal_embedding_img = modalemb

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    def init_parameters(self):
         # FIXME OpenAI CLIP did not define an init for the VisualTransformer
         # TODO experiment if default PyTorch init, below, or alternate init is best.

         # nn.init.normal_(self.class_embedding, std=self.scale)
         # nn.init.normal_(self.positional_embedding, std=self.scale)
         #
         # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
         # attn_std = self.transformer.width ** -0.5
         # fc_std = (2 * self.transformer.width) ** -0.5
         # for block in self.transformer.resblocks:
         #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
         #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
         #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
         #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
         #
         # if self.text_projection is not None:
         #     nn.init.normal_(self.text_projection, std=self.scale)
         pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        # modal_embedding_img = self.modal_embedding_img(torch.tensor(0).cuda()).to(x.dtype)# 768
        # modal_embedding_img = modal_embedding_img.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 50, -1)
        # x=x+modal_embedding_img
        
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if self.align_before:
            x,intermediates_vis = self.transformer(x)
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        if self.align_before:
            for i in range(len(intermediates_vis)):
                intermediates_vis[i] = intermediates_vis[i].permute(1, 0, 2)[:, 0, :]

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
            if self.align_before:
                intermediates_vis = [intermediate @ self.proj for intermediate in intermediates_vis]  
                return x,intermediates_vis
            
        return x

        


class TextTransformer(nn.Module):

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            ls_init_value: float = None,
            output_dim: int = 512,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            mmadapter:nn.ModuleList = None,
            mmadapter_aux:nn.ModuleList = None,
            modalemb:nn.Module = None,
            align_before:bool = False,
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.align_before = align_before

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mmadapter=mmadapter,
            mmadapter_aux=mmadapter_aux
        )
        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        # self.modal_embedding_text = modalemb

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        
#         embedding_text = self.modal_embedding_text(torch.tensor(1).cuda()).to(x.dtype)# 768
#         embedding_text = embedding_text.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 77, -1)

#         embedding_text = F.interpolate(embedding_text.unsqueeze(1), size=(77, 512), mode='bilinear', align_corners=False)
#         embedding_text = embedding_text.squeeze(1)
        
#         x=x+embedding_text
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        if self.align_before:
            x,intermediates_text = self.transformer(x, attn_mask=self.attn_mask)
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        if self.align_before:
            intermediates_text = [intermediate[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection for intermediate in intermediates_text]
            return x,intermediates_text
        
        return x

        
