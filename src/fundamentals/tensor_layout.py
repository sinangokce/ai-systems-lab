def tensor_shape(x):
    return x.shape


def tensor_stride(x):
    return x.stride()


def is_tensor_contiguous(x):
    return x.is_contiguous()
