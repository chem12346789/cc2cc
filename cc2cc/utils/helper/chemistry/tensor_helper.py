from functools import lru_cache

import opt_einsum as oe
from numba import get_num_threads, njit, prange, set_num_threads
from pyscf import lib


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


@njit(parallel=True, cache=False)
def _accumulate_w(out, source, scale, subtract):
    n0, n1, n2, n3, n4, n5 = out.shape
    for row in prange(n0 * n1 * n2):
        i = row // (n1 * n2)
        j = (row // n2) % n1
        k = row % n2
        for a in range(n3):
            for b in range(n4):
                for c in range(n5):
                    value = source[i, j, k, a, b, c]
                    if scale is not None:
                        value = value * scale
                    if subtract:
                        out[i, j, k, a, b, c] -= value
                    else:
                        out[i, j, k, a, b, c] += value


def _parallel_accumulate_w(out, source, scale, subtract):
    # Follow the destination's physical layout without copying either operand.
    axes = tuple(sorted(range(6), key=lambda axis: abs(out.strides[axis]), reverse=True))
    threads = get_num_threads()
    set_num_threads(min(threads, lib.num_threads()))
    try:
        _accumulate_w(out.transpose(axes), source.transpose(axes), scale, subtract)
    finally:
        set_num_threads(threads)


def w_einsum(
    subscripts, *tensors,
    out=None, scale=None, subtract=False,
):
    """Keep BLAS contractions and fuse parallel W scaling/accumulation."""
    result = parallel_einsum(subscripts, *tensors)
    if scale is not None and out is None:
        scaled = result * scale
        del result
        result = scaled
        del scaled
    if out is not None:
        _parallel_accumulate_w(out, result, scale, subtract)
        del result
        result = out
    return result
