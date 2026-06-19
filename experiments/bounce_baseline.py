"""
Port of run_pipeline_8s.py detect_bounces_from_trajectory (velocity-sign-change)
for use in the experimentation lab. Faithful 1:1 copy of the committed logic.
"""
import numpy as np


def detect_bounces_velocity_signchange(ball_centers, ball_speeds_px, fps, frame_height,
                                        frame_width=None):
    if len(ball_centers) < 5:
        return []
    min_gap = int(fps * 0.25)
    half_window = max(3, int(fps * 0.12))
    min_y_ratio = 0.12
    max_y_ratio = 0.93
    speed_spike_threshold = 1.8

    vy = np.full(len(ball_centers), np.nan)
    for i in range(1, len(ball_centers)):
        if ball_centers[i] is not None and ball_centers[i - 1] is not None:
            vy[i] = ball_centers[i][1] - ball_centers[i - 1][1]

    kernel = max(3, int(fps * 0.06))
    if kernel % 2 == 0:
        kernel += 1
    vy_smooth = np.copy(vy)
    half_k = kernel // 2
    for i in range(half_k, len(vy) - half_k):
        window = vy[i - half_k:i + half_k + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 2:
            vy_smooth[i] = np.mean(valid)

    bounces = []
    last_bounce_frame = -min_gap
    for i in range(half_window + 1, len(ball_centers) - half_window):
        if ball_centers[i] is None:
            continue
        cx, cy = ball_centers[i]
        y_ratio = cy / frame_height
        if y_ratio < min_y_ratio or y_ratio > max_y_ratio:
            continue
        if (i - last_bounce_frame) < min_gap:
            continue
        vy_before = np.nan
        vy_after = np.nan
        for k in range(i, max(i - half_window - 1, 0), -1):
            if not np.isnan(vy_smooth[k]) and vy_smooth[k] > 0:
                vy_before = vy_smooth[k]
                break
        for k in range(i, min(i + half_window + 1, len(vy_smooth))):
            if not np.isnan(vy_smooth[k]) and vy_smooth[k] < 0:
                vy_after = vy_smooth[k]
                break
        if np.isnan(vy_before) or np.isnan(vy_after):
            continue
        speed_win = max(3, int(fps * 0.15))
        speeds_before = [ball_speeds_px[j] for j in range(max(0, i - speed_win), i)
                         if ball_speeds_px[j] > 0]
        speeds_after = [ball_speeds_px[j] for j in range(i, min(len(ball_speeds_px), i + speed_win))
                        if ball_speeds_px[j] > 0]
        avg_before = np.mean(speeds_before) if speeds_before else 0
        avg_after = np.mean(speeds_after) if speeds_after else 0
        speed_ratio = avg_after / max(avg_before, 0.1)
        if speed_ratio > speed_spike_threshold:
            continue
        bounces.append((i, cx, cy))
        last_bounce_frame = i
    return bounces
