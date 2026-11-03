from functools import lru_cache

import opt_einsum as oe


@lru_cache(maxsize=128)
def _contract_expression(subscripts, shapes):
    return oe.contract_expression(
        subscripts,
        *shapes,
        optimize="optimal",
        use_blas=True,
    )


def parallel_einsum(subscripts, *tensors):
    """Contract NumPy arrays using cached optimal plans and eligible BLAS kernels.

    Only plans are cached, never input arrays or results. BLAS thread counts
    remain controlled by the installed BLAS runtime.
    """
    shapes = tuple(tensor.shape for tensor in tensors)
    expression = _contract_expression(subscripts, shapes)
    return expression(*tensors, backend="numpy")
