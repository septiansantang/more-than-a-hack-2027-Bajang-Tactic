import numpy as np

# Batas video frame resmi dari DATA.md (indeks dimulai dari 0)
FIRST_HALF_START = 11185
FIRST_HALF_END = 90651
SECOND_HALF_START = 110808
SECOND_HALF_KICKOFF = 113742
SECOND_HALF_END = 190197


def get_graded_mask(n_frames=199319):
    """
    Membuat mask boolean untuk frame yang dinilai (graded = True)
    dan frame di luar pertandingan yang diberi label X (graded = False).
    """
    graded = np.zeros(n_frames, dtype=bool)
    graded[FIRST_HALF_START : FIRST_HALF_END + 1] = True
    graded[SECOND_HALF_START : SECOND_HALF_END + 1] = True
    return graded


def compute_possession_probs(features_df, sigma_d=4.5):
    """
    Menghitung matriks probabilitas [P(H), P(A), P(D)] berukuran (N, 3)
    murni dari fitur jarak pemain, kerumunan 5m, kinematika bola, dan laju pemain.
    """
    n_frames = len(features_df)
    probs = np.zeros((n_frames, 3), dtype=np.float32)

    ball_det = features_df["ball_detected"].values == 1
    d_h = features_df["home_min_distance"].values
    d_a = features_df["away_min_distance"].values
    h_5m = np.nan_to_num(features_df["home_players_5m"].values, nan=0.0)
    a_5m = np.nan_to_num(features_df["away_players_5m"].values, nan=0.0)

    # Kecepatan kolektif pemain
    h_speed = features_df["home_median_speed"].values
    a_speed = features_df["away_median_speed"].values
    both_valid = np.isfinite(h_speed) & np.isfinite(a_speed)
    collective_speed = np.where(
        both_valid,
        0.5 * (np.nan_to_num(h_speed) + np.nan_to_num(a_speed)),
        np.where(
            np.isfinite(h_speed),
            h_speed,
            np.where(np.isfinite(a_speed), a_speed, 1.5)
        )
    )

    kernel = np.ones(25, dtype=np.float32) / 25.0
    collective_speed_smooth = np.convolve(
        collective_speed.astype(np.float32), kernel, mode="same"
    )

    out_of_bounds = (
        features_df["ball_out_of_bounds"].values == 1
        if "ball_out_of_bounds" in features_df.columns
        else np.zeros(n_frames, dtype=bool)
    )
    ball_speed = (
        features_df["ball_speed"].values
        if "ball_speed" in features_df.columns
        else np.zeros(n_frames, dtype=np.float32)
    )
    ball_angle = (
        features_df["ball_angle_diff"].values
        if "ball_angle_diff" in features_df.columns
        else np.zeros(n_frames, dtype=np.float32)
    )

    # 1. Skor kontrol jarak eksponensial + nilai dasar agar umpan jauh tidak jatuh ke D
    valid_h = np.isfinite(d_h)
    valid_a = np.isfinite(d_a)

    score_h = np.where(valid_h, 0.04 + np.exp(-(d_h**2) / (2.0 * sigma_d**2)), 0.15)
    score_a = np.where(valid_a, 0.04 + np.exp(-(d_a**2) / (2.0 * sigma_d**2)), 0.15)

    # Tambahan skor bagi tim yang lebih dekat ke bola
    closer_h = valid_h & valid_a & (d_h < d_a)
    closer_a = valid_h & valid_a & (d_a < d_h)
    score_h = np.where(closer_h, score_h + 0.05, score_h)
    score_a = np.where(closer_a, score_a + 0.05, score_a)

    score_h += 0.08 * np.minimum(h_5m, 3.0)
    score_a += 0.08 * np.minimum(a_5m, 3.0)

    # 2. Skor Dead Ball (D) dari geometri & kinematika (< 1.05 m/s)
    score_d = np.full(n_frames, 0.03, dtype=np.float32)

    pre_second_half = np.zeros(n_frames, dtype=bool)
    pre_second_half[SECOND_HALF_START:SECOND_HALF_KICKOFF] = True
    score_d = np.where(pre_second_half | out_of_bounds, 3.50, score_d)

    slow_play_no_ball = (~ball_det) & (collective_speed_smooth < 1.05)
    score_d = np.where(slow_play_no_ball, np.maximum(score_d, 1.50), score_d)

    stationary_uncontested = (
        ball_det
        & np.isfinite(ball_speed) & (ball_speed < 0.4)
        & (d_h > 4.0) & (d_a > 4.0)
        & (collective_speed_smooth < 1.15)
    )
    score_d = np.where(stationary_uncontested, np.maximum(score_d, 1.35), score_d)

    raw_stack = np.vstack([score_h, score_a, score_d]).T
    row_sums = np.sum(raw_stack, axis=1, keepdims=True)
    probs = raw_stack / np.maximum(row_sums, 1e-6)

    # 3. Pass-Through Suppression & Forward-Fill saat bola hilang di permainan aktif
    is_fast_straight_pass = (
        ball_det
        & np.isfinite(ball_speed) & (ball_speed > 7.5)
        & np.isfinite(ball_angle) & (ball_angle < 15.0)
        & (d_h > 1.5) & (d_a > 1.5)
    )
    active_missing = (~ball_det) & (~pre_second_half) & (collective_speed_smooth >= 1.05)
    hold_possession = active_missing | is_fast_straight_pass

    last_h, last_a = 0.5, 0.5
    for i in range(n_frames):
        if not hold_possession[i] and ball_det[i] and not out_of_bounds[i]:
            s_ha = probs[i, 0] + probs[i, 1]
            if s_ha > 1e-4:
                last_h = probs[i, 0] / s_ha
                last_a = probs[i, 1] / s_ha
        elif hold_possession[i]:
            rem = max(0.0, 1.0 - probs[i, 2])
            probs[i, 0] = rem * last_h
            probs[i, 1] = rem * last_a

    # Normalisasi akhir agar total probabilitas setiap baris bernilai 1.0
    final_sums = np.sum(probs, axis=1, keepdims=True)
    probs = probs / np.maximum(final_sums, 1e-6)

    return probs.astype(np.float32)


def predict_possession_labels(probs, graded_mask):
    """
    Mengonversi matriks probabilitas (N, 3) menjadi array label 'H', 'A', 'D', 'X'.
    """
    label_map = np.array(["H", "A", "D"])
    best_idx = np.argmax(probs, axis=1)
    labels = label_map[best_idx]
    labels[~graded_mask] = "X"
    return labels