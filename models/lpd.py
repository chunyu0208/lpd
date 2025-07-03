import numpy as np
import torch
import torch.nn as nn

from functools import partial

from models.block import Block
from models.mask import generate_mask_mod
from models.sampling import sample
from torch.nn.attention.flex_attention import create_mask
from torch.utils.checkpoint import checkpoint


class LPD(nn.Module):
    def __init__(self, img_size=256, vqgan_stride=16,
                 decoder_embed_dim=1024, decoder_depth=32, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 grad_checkpointing=False,
                 vqgan_vocab_size=16384
                 ):
        super().__init__()

        self.img_size = img_size
        self.vqgan_stride = vqgan_stride
        self.seq_h = self.seq_w = img_size // vqgan_stride 
        self.seq_len = self.seq_h * self.seq_w
        self.num_classes = class_num
        self.label_drop_prob = label_drop_prob
        self.grad_checkpointing = grad_checkpointing

        self.class_emb = nn.Embedding(class_num, decoder_embed_dim)
        self.token_emb = nn.Embedding(vqgan_vocab_size, decoder_embed_dim)
        self.fake_latent = nn.Parameter(torch.zeros(1, decoder_embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, decoder_embed_dim))
        self.z_proj_ln = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])
        
        self.norm = norm_layer(decoder_embed_dim)
        self.head = nn.Linear(decoder_embed_dim, vqgan_vocab_size, bias=False)
        self.criterion = nn.CrossEntropyLoss()

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.token_emb.weight, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def sample_orders(self, bsz):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def forward_decoder(self, x, class_embedding, group_sizes, orders):
        bsz, seq_len, embed_dim = x.shape

        x = torch.cat([torch.zeros(bsz, 1, embed_dim, device=x.device), x], dim=1)

        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :1] = class_embedding.unsqueeze(1)
        masked_tokens = self.mask_token.expand(bsz, seq_len+1, embed_dim)

        x = x + self.pos_embed
        x = self.z_proj_ln(x)
        masked_tokens = masked_tokens + self.pos_embed
        masked_tokens = self.z_proj_ln(masked_tokens)

        orders = torch.cat([torch.zeros(bsz, 1, device=orders.device, dtype=orders.dtype), orders+1], dim=1)
        x = torch.gather(x, dim=1, index=orders.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        masked_tokens = torch.gather(masked_tokens, dim=1, index=orders.unsqueeze(-1).expand(-1, -1, masked_tokens.shape[2]))
        
        counts = group_sizes[1:] - group_sizes[:-1]
        x = torch.cat([x[:, :-counts[-1]], masked_tokens[:, counts[0]:]], dim=1)
        mask_mod = generate_mask_mod(group_sizes)
        mask = create_mask(mask_mod, B=None, H=None, Q_LEN=x.shape[1], KV_LEN=x.shape[1], device=x.device)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.blocks:
                x = checkpoint(block, x, mask)
        else:
            for block in self.blocks:
                x = block(x, mask)
        
        x = self.norm(x)
        x = x[:, -seq_len:]

        return x

    def forward_loss(self, output, target):
        bsz, seq_len = target.shape
        loss = self.criterion(output.view(bsz * seq_len, -1), target.view(-1))
        return loss

    def forward(self, x, labels, group_sizes):
        class_embedding = self.class_emb(labels)

        target = x.clone().detach()
        x = self.token_emb(x)

        orders = self.sample_orders(bsz=x.size(0))
        target = torch.gather(target, dim=1, index=orders)

        group_sizes = eval(group_sizes) if isinstance(group_sizes, str) else group_sizes
        group_sizes = torch.tensor(group_sizes, dtype=torch.int, device=x.device)
        group_sizes = torch.cumsum(group_sizes, dim=0)
        group_sizes = torch.cat([torch.tensor([0, 1], dtype=torch.int, device=x.device), group_sizes+1])

        hidden_output = self.forward_decoder(x, class_embedding, group_sizes, orders)
        output = self.head(hidden_output)

        loss = self.forward_loss(output=output, target=target)

        return loss

    def enable_kv_cache(self, max_bs, max_seq_len, n_cond_tokens: int = 1, device = "cuda"):
        max_seq_len = n_cond_tokens + max_seq_len
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.float32
        for block in self.blocks:
            block.attn.kv_cache = True
            block.attn.enable_kv_cache(max_bs, max_seq_len, device, dtype)

    def disable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = False
            block.attn.disable_kv_cache()
    
    def sample_tokens(self, bsz, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, attn_mask = None, top_k=0, top_p=1.0, group_sizes = None, orders: torch.Tensor = None):
        device = "cuda"
        tokens = torch.zeros(bsz, self.seq_len, dtype=torch.int64, device=device)

        orders_original = orders.clone()
        orders = torch.cat([torch.zeros(bsz, 1, device=orders.device, dtype=orders.dtype), orders+1], dim=1)
        pos_embed = torch.gather(self.pos_embed.clone().repeat(bsz, 1, 1), dim=1, index=orders.unsqueeze(-1).expand(-1, -1, self.pos_embed.shape[2]))
 
        if labels is not None:
            class_embedding = self.class_emb(labels)
        else:
            class_embedding = self.fake_latent.repeat(bsz, 1)

        if not cfg == 1.0:
            class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
            pos_embed = torch.cat([pos_embed, pos_embed], dim=0)
            bsz = bsz * 2

        self.enable_kv_cache(max_bs=bsz, max_seq_len=tokens.shape[1], n_cond_tokens=1, device=tokens.device)

        group_sizes = torch.tensor(eval(group_sizes) if isinstance(group_sizes, str) else group_sizes, dtype=torch.int, device=device)
        group_sizes_cumsum = torch.cumsum(group_sizes, dim=0)
        group_sizes_cumsum = torch.cat([torch.tensor([0, 1], dtype=torch.int, device=device), group_sizes_cumsum+1])

        for i in range(len(group_sizes)):
            if i == 0:
                previous_token = class_embedding.unsqueeze(1)
            else:
                previous_token = self.token_emb(tokens[:, group_sizes_cumsum[i]-1 : group_sizes_cumsum[i+1]-1])
                if not cfg == 1.0:
                    previous_token = torch.cat([previous_token, previous_token], dim=0)

            x_mask = self.mask_token.repeat(bsz, group_sizes[i], 1)

            x = torch.cat([previous_token, x_mask], dim = 1) 
            x = x + pos_embed[:, group_sizes_cumsum[i]:group_sizes_cumsum[i+2], :]
            x = self.z_proj_ln(x)

            input_pos = torch.arange(group_sizes_cumsum[i], group_sizes_cumsum[i+2])
            
            for block in self.blocks:
                x = block.forward_inference(x, input_pos, attn_mask)

            x = self.norm(x)
            x = x[:, -group_sizes[i]:]
            logits = self.head(x)

            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * torch.linspace(group_sizes_cumsum[i+1] - 1, group_sizes_cumsum[i+1] + logits.shape[1] - 2, logits.shape[1], device=device) / (group_sizes_cumsum[-1] - 1)
            elif cfg_schedule == "constant":
                cfg_iter = torch.tensor([cfg], device=device)
            else:
                raise NotImplementedError
            if not cfg == 1.0:
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                B, N, C = cond_logits.shape
                logits = uncond_logits + cfg_iter.unsqueeze(0).unsqueeze(-1).expand(B, N, C) * (cond_logits - uncond_logits)
            else:
                B, N, C = logits.shape

            sampled_token = sample(logits.reshape(B*N, 1, C), temperature=temperature, top_k=top_k, top_p=top_p)[0]
            sampled_token = sampled_token.reshape(B, N)
            tokens[:, group_sizes_cumsum[i+1]-1:group_sizes_cumsum[i+2]-1] = sampled_token

        tokens_ = torch.zeros_like(tokens)
        tokens_.scatter_(dim=1, index=orders_original, src=tokens)
        tokens = tokens_

        self.disable_kv_cache()

        return tokens.long()


def lpd_l(**kwargs):
    model = LPD(
        decoder_embed_dim=1024, decoder_depth=24, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def lpd_xl(**kwargs):
    model = LPD(
        decoder_embed_dim=1280, decoder_depth=36, decoder_num_heads=20,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def lpd_xxl(**kwargs):
    model = LPD(
        decoder_embed_dim=1536, decoder_depth=48, decoder_num_heads=24,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
