def tuple_to_device(tup, device):
    return tuple(d.to(device) for d in tup)


def dict_to_device(dict, device):
    return {k: v.to(device) for k, v in dict.items()}


def squeeze_dict(dict):
    return {k: v.squeeze(0) for k, v in dict.items()}
