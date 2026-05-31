import os
import cv2
import numpy as np

IR_RED_RATIO = 1.25
FORCE_GRAYSCALE = os.getenv("YUV_FORCE_GRAYSCALE", "0") == "1"

EMA_ALPHA_GAINS = 0.18

def compute_gray_world_gains(img, max_gain=2.0, eps=1e-6):
    img_f = img.astype(np.float32)
    means = img_f.reshape(-1, 3).mean(axis=0)
    overall = means.mean()
    gains = overall / (means + eps)
    gains = np.clip(gains, 1.0 / max_gain, max_gain)
    return gains


def preprocess_frame(frame, prev_gains):
    if frame is None:
        return None, prev_gains

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # channel swap (your camera quirk fix)
    chosen = frame[..., ::-1]

    # gray world WB with smoothing
    gains_raw = compute_gray_world_gains(chosen)

    if prev_gains is None:
        prev_gains = gains_raw
    else:
        prev_gains = prev_gains * (1.0 - EMA_ALPHA_GAINS) + gains_raw * EMA_ALPHA_GAINS

    wb = (chosen.astype(np.float32) * prev_gains)
    wb = np.clip(wb, 0, 255).astype(np.uint8)

    # IR detection fallback
    b_mean, g_mean, r_mean = wb.reshape(-1, 3).mean(axis=0)
    gb_mean = (g_mean + b_mean) / 2.0

    if FORCE_GRAYSCALE or (r_mean > gb_mean * IR_RED_RATIO):
        gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
        proc = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        proc = wb

    return proc, prev_gains