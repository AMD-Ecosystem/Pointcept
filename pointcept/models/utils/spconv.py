"""
SpConv Backend Selection

Sparse convolution models are written against the spconv 2.x API. spconv ships
prebuilt CUDA kernels and has no AMD GPU build, so on a ROCm PyTorch the same API
is taken from spconv-triton, a drop-in reimplementation of spconv 2.x in Triton
(`pip install spconv-triton`, refer to `README.md`), falling back to spconv when
spconv-triton cannot be imported.
"""

import warnings

import torch

if torch.version.hip is None:
    import spconv.pytorch as spconv
else:
    try:
        import spconv_triton.pytorch as spconv
    except ImportError as triton_error:
        try:
            import spconv.pytorch as spconv
        except ImportError as spconv_error:
            raise ImportError(
                "Sparse convolution on a ROCm PyTorch needs spconv-triton "
                "(`pip install spconv-triton`) or spconv, and neither can be imported "
                f"(spconv_triton: {triton_error}; spconv: {spconv_error})"
            ) from spconv_error
        if not (
            isinstance(triton_error, ModuleNotFoundError)
            and triton_error.name == "spconv_triton"
        ):
            warnings.warn(
                f"spconv-triton is installed but failed to import ({triton_error}), "
                "using spconv instead"
            )

__all__ = ["spconv"]
