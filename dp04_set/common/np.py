# np.py : NumPy / CuPy 自動切り替え
import numpy as np

try:
    import cupy as cp
    gpu_enable = True
except ImportError:
    gpu_enable = False

def asnumpy(x):
    """cupy → numpy 変換"""
    if isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)

def asarray(x):
    """numpy / cupy 共通変換"""
    if gpu_enable:
        return cp.asarray(x)
    return np.asarray(x)

def get_array_module(x):
    """x に応じて numpy または cupy モジュールを返す"""
    if isinstance(x, np.ndarray):
        return np
    return cp

# GPU 有効時は CuPy モジュールを、そうでなければ NumPy を返す
if gpu_enable:
    print(">>> GPU mode (CuPy) <<<")
else:
    print(">>> CPU mode (NumPy) <<<")
