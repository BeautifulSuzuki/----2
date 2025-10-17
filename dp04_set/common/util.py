# util.py : 汎用ユーティリティ関数
import numpy as np
from common.np import get_array_module

# --- 勾配の大きさを制限 ---
def clip_grads(grads, max_norm):
    """
    勾配ベクトル全体のノルムが max_norm を超えないようにスケーリングする
    （勾配爆発防止）
    """
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        rate = max_norm / (total_norm + 1e-6)
        for grad in grads:
            grad *= rate

# --- one-hot表現への変換 ---
def to_one_hot(y, num_classes=None):
    """整数ラベル y を one-hot ベクトルに変換"""
    xp = get_array_module(y)
    if num_classes is None:
        num_classes = int(xp.max(y)) + 1
    one_hot = xp.zeros((y.size, num_classes), dtype=xp.float32)
    for idx, val in enumerate(y):
        one_hot[idx, int(val)] = 1
    return one_hot
