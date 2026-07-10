import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import PatchEmbed, trunc_normal_
from src.utils.model_utils import load_constant
from src.models.forecast.layers.pos_embed import get_2d_sincos_pos_embed


class ClimaX(nn.Module):
    def __init__(
        self,
        default_vars,
        img_size=[64, 128],
        patch_size=4,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.2,
        drop_rate=0.2,
        const_dir="../../data/train_pred",
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.c = len(self.default_vars)
        self.h = self.img_size[0] // patch_size
        self.w = self.img_size[1] // patch_size
        self.embed_dim = embed_dim
        self.constant = torch.from_numpy(load_constant(const_dir))

        self.var_map = self.create_var_map()
        self.patch_embed = PatchEmbed(img_size, patch_size, len(default_vars), embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        self.lead_time_embed = nn.Linear(1, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop_rate,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop_rate,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, len(self.default_vars) * patch_size ** 2)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

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

    def unpatchify(self, x: torch.Tensor):
        x = x.reshape(shape=(x.shape[0], self.h, self.w, self.patch_size, self.patch_size, self.c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.c, self.h * self.patch_size, self.w * self.patch_size))
        return imgs

    def forward_encoder(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape(-1, self.h, self.w, self.embed_dim)
        x = x.flatten(1, 2)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, x, variables, out_variables):
        constant = torch.repeat_interleave(self.constant, x.shape[0], dim=0).to(x.device, dtype=x.dtype)
        x = torch.concat([constant, x], dim=1)

        out_transformers = self.forward_encoder(x)

        dec_input = self.decoder_embed(out_transformers)
        for blk in self.decoder_blocks:
            dec_input = blk(dec_input, out_transformers)
        dec_input = self.decoder_norm(dec_input)

        preds = self.head(dec_input)
        preds = preds.reshape(-1, self.h, self.w, preds.shape[-1])

        preds = self.unpatchify(preds)
        out_var_ids = self.get_var_ids(out_variables, preds.device)
        preds = preds[:, out_var_ids]

        return preds