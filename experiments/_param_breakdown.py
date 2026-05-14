import torch, math

ck    = torch.load('checkpoints/07_best_model.pt', map_location='cpu', weights_only=True)
state = ck['model_state']

groups = {'embedding': 0, 'lm_head': 0, 'ns_physics': 0, 'mlp': 0, 'norm': 0, 'other': 0}

print(f"{'Name':<55} {'Params':>10}")
print('-' * 68)
for name, tensor in state.items():
    n = tensor.numel()
    print(f"{name:<55} {n:>10,}")
    if 'token_emb' in name or 'pos_emb' in name:
        groups['embedding'] += n
    elif 'lm_head' in name:
        groups['lm_head'] += n
    elif any(x in name for x in ['log_nu','log_dt','log_alpha','log_p_scale']):
        groups['ns_physics'] += n
    elif 'mlp' in name:
        groups['mlp'] += n
    elif 'norm' in name:
        groups['norm'] += n
    else:
        groups['other'] += n

total = sum(groups.values())
print()
print('=' * 68)
print('GRUP OZETI:')
for k, v in groups.items():
    pct = 100 * v / total if total else 0
    print(f"  {k:<20} {v:>10,}  ({pct:.1f}%)")
print(f"  {'TOPLAM':<20} {total:>10,}")
