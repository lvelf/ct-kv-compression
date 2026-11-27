"""Answer-shift calibration (accuracy round 2, guideline section 3). Scores are s = -(mean answer-token loss), larger is
better. b_t(y) = s_Ct(null, y) - s_C0(null, y);  s*(x, y) = s_Ct(x, y) - lam * b_t(y);  prediction = first argmax."""
import numpy as np


def shift(null_loss_t, null_loss_0):
    """b_t per null item from the stored FP32 losses (loss = -s, so b = loss_0 - loss_t)."""
    return np.asarray(null_loss_0, dtype=np.float64) - np.asarray(null_loss_t, dtype=np.float64)


def calibrated(real_loss, b, lam):
    """s* for one question: real_loss and b are aligned per option."""
    return -np.asarray(real_loss, dtype=np.float64) - lam * np.asarray(b, dtype=np.float64)


def argbest(scores):
    return int(np.argmax(scores))  # first maximum, like min() over losses in the original evaluator


def self_test():
    real = [1.0, 1.2]                       # s = [-1.0, -1.2]: option 0 wins before correction
    assert argbest(calibrated(real, [0.0, 0.0], 0.5)) == 0
    assert argbest(calibrated(real, [5.0, -5.0], 0.0)) == 0                 # lambda = 0: unchanged
    b = shift([0.2, 1.0], [1.0, 1.0])                                       # option 0 gained +0.8 under null
    assert np.allclose(b, [0.8, 0.0])
    assert argbest(calibrated(real, b, 0.5)) == 1                           # the positively shifted option is penalised
    assert argbest(calibrated(real, -b, 0.5)) == 0                          # wrong sign would favour it instead
    assert np.allclose(shift([0.3, 0.7], [0.3, 0.7]), 0.0)                  # C_t = C_0: zero shift
    return True
