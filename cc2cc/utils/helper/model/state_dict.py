import torch

STATE_DICT_PREFIXES = ("_orig_mod.",)


def normalize_state_dict(state_dict: dict[str, torch.Tensor]):
    normalized_state_dict = {}
    for key, value in state_dict.items():
        normalized_key = key
        prefix_removed = True
        while prefix_removed:
            prefix_removed = False
            for prefix in STATE_DICT_PREFIXES:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
                    prefix_removed = True
        normalized_state_dict[normalized_key] = value
    return normalized_state_dict
