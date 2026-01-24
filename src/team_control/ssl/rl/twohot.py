import torch
import numpy as np


def two_hot_encode(value, num_bins, min_value, max_value):
    bin_width = (max_value - min_value) / (num_bins - 1)
    pos = (value - min_value) / bin_width

    lower = int(np.floor(pos))
    upper = min(lower + 1, num_bins - 1)

    upper_w = pos - lower
    lower_w = 1.0 - upper_w

    vec = np.zeros(num_bins, dtype=np.float32)
    vec[max(0, lower)] += lower_w
    vec[upper] += upper_w

    return torch.tensor(vec, dtype=torch.float32)


def two_hot_decode(dist, num_bins, min_value, max_value):
    bin_values = torch.linspace(min_value, max_value, num_bins, device=dist.device)
    return (dist * bin_values).sum(dim=-1)


# def two_hot_encode(value, num_bins, min_value, max_value):
#     # value = torch.as_tensor(value, dtype=torch.float32).to(device)
#     bin_width = (max_value - min_value) / (num_bins - 1)

#     pos = (value - min_value) / bin_width
#     lower = torch.floor(torch.tensor(pos)).long()
#     upper = lower + 1

#     upper_weight = pos - lower.float()
#     lower_weight = 1.0 - upper_weight

#     lower = torch.clamp(lower, 0, num_bins - 1)
#     upper = torch.clamp(upper, 0, num_bins - 1)

#     two_hot = torch.zeros(1, num_bins).to(device)
#     two_hot.scatter_(1, lower.unsqueeze(1), lower_weight.unsqueeze(1))
#     two_hot.scatter_(1, upper.unsqueeze(1), upper_weight.unsqueeze(1))

#     return two_hot

if __name__ == '__main__':
    # TODO: add some testing here for the twohot so that we are sure this essential function works correctly
    print("some test on twohot encode/decode")
