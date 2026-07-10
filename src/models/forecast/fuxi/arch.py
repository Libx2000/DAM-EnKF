import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import trunc_normal_
from src.utils.model_utils import load_constant
from src.models.forecast.layers.pos_embed import get_2d_sincos_pos_embed


class FuXi(nn.Module):
    def __init__(
        self,
        default_vars,
        img_size=[64, 128],
        window_size=4,
        patch_size=4,
        down_times=0,
        embed_dim=1024,
        num_heads=16,
        depths=[6, 6],
        mlp_ratio=4.0,
        drop_path=0.2,
        drop_rate=0.2,
        attn_drop=0.0,
        const_dir="../../data/train_pred",
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.default_vars = default_vars
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.c = len(self.default_vars)
        self.h = self.img_size[0] // patch_size
        self.w = self.img_size[1] // patch_size
        self.embed_dim = embed_dim
        self.constant = torch.from_numpy(load_constant(const_dir))

        self.var_map = self.create_var_map()

        self.patch_embed = nn.Conv2d(len(default_vars), embed_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.h, self.w), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer, depth in enumerate(depths):
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]) + i],
                    attn_drop=attn_drop,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ])
            self.layers.append(layer)

        self.norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, len(self.default_vars) * patch_size ** 2, kernel_size=1),
            nn.PixelShuffle(patch_size),
        )

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[1],
            self.pos_embed.shape[2],
            self.pos_embed.shape[3],
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().permute(1, 0).reshape(1, self.embed_dim, self.h, self.w))

        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_map(self):
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_map

    def get_var_ids(self, vars, device):
        ids = [self.var_map[var] for var in vars]
        return torch.tensor(ids, device=device)

    def forward(self, x, variables, out_variables):
        constant = torch.repeat_interleave(self.constant, x.shape[0], dim=0).to(x.device, dtype=x.dtype)
        x = torch.concat([constant, x], dim=1)

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            for blk in layer:
                x = blk(x)

        x = self.norm(x)
        preds = self.head(x)

        out_var_ids = self.get_var_ids(out_variables, preds.device)
        preds = preds[:, out_var_ids]

        return preds


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        attn_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
        self.window_size = window_size
        self.shift_size = shift_size

    def forward(self, x):
        B, C, H, W = x.shape

        shortcut = x
        x = self.norm1(x.permute(0, 2, 3, 1)).reshape(B, H * W, C)

        if self.shift_size > 0:
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        x, _ = self.attn(x, x, x)

        if self.shift_size > 0:
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        x = shortcut + self.drop_path(x.reshape(B, H, W, C).permute(0, 3, 1, 2))

        shortcut = x
        x = self.norm2(x.permute(0, 2, 3, 1)).reshape(B, H * W, C)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x.reshape(B, H, W, C).permute(0, 3, 1, 2))

        return x