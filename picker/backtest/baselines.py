"""三個對照基準（規格 8.3）：隨機、純動能、基準指數買進持有。研究用途，非投資建議。"""
import numpy as np


def momentum_score(ranked):
    return ranked["ret_20"]


def random_score(panel, seed):
    rng = np.random.default_rng(seed)
    return panel["ret_20"].notna().astype(float) * rng.random(len(panel))
