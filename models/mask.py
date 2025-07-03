import torch

from torch.nn.attention.flex_attention import _mask_mod_signature

# Use flex_attention to generate training mask
def generate_mask_mod(group_sizes) -> _mask_mod_signature:

    counts = group_sizes[1:] - group_sizes[:-1]
    context_len = counts[:-1].sum()
    device = group_sizes.device

    def _group_sizes_to_group_ids_tensor(group_sizes):
        ids = []
        num_blocks = 2*(len(counts)-2)+2
        num_blocks_half = num_blocks // 2

        for i in range(num_blocks):
            if i < num_blocks_half:
                ids.extend([i*2] * counts[i])
            else:
                ids.extend([(i-num_blocks_half)*2+1] * counts[i-num_blocks_half+1])

        pad_length = 128 - len(ids) % 128
        ids.extend(torch.arange(-1, -1-pad_length, -1, device=device))

        return torch.tensor(ids, device=device, dtype=torch.int32)

    group_id = _group_sizes_to_group_ids_tensor(group_sizes)

    def mask_mod(b, h, q_idx, kv_idx):
        same_group = group_id[q_idx] == group_id[kv_idx]
        causal = (group_id[kv_idx] < group_id[q_idx])
        context = (kv_idx < context_len)

        return same_group | (causal & context)

    return mask_mod