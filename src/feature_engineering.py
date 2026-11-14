
import numpy as np

def calculate_player_ball_distances(
    record,
    pitch_length=105.3,
    pitch_width=68.0
):
    """
    Menghitung jarak setiap pemain ke bola dalam meter.

    Koordinat ATD dinormalisasi 0-1, sehingga:
    X dikalikan panjang lapangan.
    Y dikalikan lebar lapangan.

    Mengembalikan array jarak untuk setiap track slot.
    Nilai NaN berarti jarak tidak dapat dihitung.
    """
    players = record["players"]
    ball = record["ball"]

    distances = np.full(len(players), np.nan)

    if ball is None:
        return distances

    ball_x = ball["x"] * pitch_length
    ball_y = ball["y"] * pitch_width

    if not np.isfinite(ball_x) or not np.isfinite(ball_y):
        return distances

    for index, player in enumerate(players):
        if player is None:
            continue

        player_x = player["x"] * pitch_length
        player_y = player["y"] * pitch_width

        if not np.isfinite(player_x) or not np.isfinite(player_y):
            continue

        dx = player_x - ball_x
        dy = player_y - ball_y

        distances[index] = np.sqrt(dx**2 + dy**2)

    return distances



def analyze_nearest_player(distances, team_mapping):
    """
    Menganalisis pemain terdekat dengan bola
    berdasarkan jarak dan mapping tim.
    """

    valid_indices = np.where(np.isfinite(distances))[0]

    if len(valid_indices) == 0:
        return {
            "nearest_player": None,
            "nearest_distance": None,
            "nearest_team": None,
            "home_players": 0,
            "away_players": 0
        }

    nearest_index = valid_indices[
        np.argmin(distances[valid_indices])
    ]

    return {
        "nearest_player": int(nearest_index),
        "nearest_distance": float(distances[nearest_index]),
        "nearest_team": int(team_mapping[nearest_index]),
        "home_players": int(
            np.sum(team_mapping[valid_indices] == 0)
        ),
        "away_players": int(
            np.sum(team_mapping[valid_indices] == 1)
        )
    }



def calculate_team_ball_distances(distances, team_mapping):
    """
    Menghitung jarak minimum pemain Home dan Away
    terhadap bola dalam meter.
    
    team_mapping:
        0 = Home
        1 = Away
    """

    valid_mask = np.isfinite(distances)

    home_mask = valid_mask & (team_mapping == 0)
    away_mask = valid_mask & (team_mapping == 1)

    home_distances = distances[home_mask]
    away_distances = distances[away_mask]

    home_min = (
        float(np.min(home_distances))
        if len(home_distances) > 0 else np.nan
    )

    away_min = (
        float(np.min(away_distances))
        if len(away_distances) > 0 else np.nan
    )

    return {
        "home_min_distance": home_min,
        "away_min_distance": away_min,
        "distance_difference": (
            abs(home_min - away_min)
            if np.isfinite(home_min) and np.isfinite(away_min)
            else np.nan
        )
    }


def count_players_near_ball(distances, team_mapping, radii=(5, 10)):
    """
    Menghitung jumlah pemain Home dan Away
    dalam beberapa radius dari bola.

    distances: array jarak pemain ke bola dalam meter.
    team_mapping: 0 = Home, 1 = Away.
    radii: daftar radius pengamatan dalam meter.
    """

    if len(distances) != len(team_mapping):
        raise ValueError("Jumlah distances dan team_mapping tidak sama.")

    results = {}

    for radius in radii:
        home_count = np.sum(
            np.isfinite(distances)
            & (team_mapping == 0)
            & (distances <= radius)
        )

        away_count = np.sum(
            np.isfinite(distances)
            & (team_mapping == 1)
            & (distances <= radius)
        )

        results[radius] = {
            "home_count": int(home_count),
            "away_count": int(away_count),
            "total_count": int(home_count + away_count)
        }

    return results


def calculate_team_speed_features(record, team_mapping):
    """
    Menghitung fitur kecepatan pemain untuk setiap tim
    pada satu frame.

    0 = Home
    1 = Away

    Kecepatan yang tidak tersedia tidak dianggap nol.
    """

    home_speeds = []
    away_speeds = []

    for index, player in enumerate(record["players"]):

        if player is None:
            continue

        speed = player["speed"]

        if not np.isfinite(speed):
            continue

        if team_mapping[index] == 0:
            home_speeds.append(speed)
        else:
            away_speeds.append(speed)

    return {
        "home_median_speed": (
            float(np.median(home_speeds))
            if home_speeds else np.nan
        ),
        "away_median_speed": (
            float(np.median(away_speeds))
            if away_speeds else np.nan
        ),
        "home_valid_speed_count": len(home_speeds),
        "away_valid_speed_count": len(away_speeds)
    }



def calculate_ball_kinematics(
    record,
    previous_record=None,
    pitch_length=105.3,
    pitch_width=68.0,
    fps=25
):
    """
    Mengekstrak posisi bola (meter), status keluar lapangan,
    serta kecepatan bola dari perpindahan antar-frame berurutan.
    """
    ball = record["ball"]

    if ball is None or not np.isfinite(ball["x"]) or not np.isfinite(ball["y"]):
        return {
            "ball_x": np.nan,
            "ball_y": np.nan,
            "ball_out_of_bounds": 0,
            "ball_displacement": np.nan,
            "ball_speed": np.nan
        }

    ball_x = float(ball["x"] * pitch_length)
    ball_y = float(ball["y"] * pitch_width)

    # Cek apakah bola keluar batas lapangan (0 s.d. pitch_length / pitch_width)
    out_of_bounds = int(
        ball_x < 0.0 or ball_x > pitch_length or
        ball_y < 0.0 or ball_y > pitch_width
    )

    displacement = np.nan
    derived_speed = np.nan

    if (
        previous_record is not None
        and record["frame_id"] == previous_record["frame_id"] + 1
        and previous_record["ball"] is not None
    ):
        prev_ball = previous_record["ball"]
        prev_x = prev_ball["x"] * pitch_length
        prev_y = prev_ball["y"] * pitch_width

        if np.isfinite(prev_x) and np.isfinite(prev_y):
            dx = ball_x - prev_x
            dy = ball_y - prev_y
            displacement = float(np.sqrt(dx**2 + dy**2))
            derived_speed = float(displacement * fps)

    return {
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_out_of_bounds": out_of_bounds,
        "ball_displacement": displacement,
        "ball_speed": derived_speed
    }


def extract_frame_features(
    record,
    team_mapping,
    previous_record=None,
    pitch_length=105.3,
    pitch_width=68.0,
    radii=(5, 10)
):
    """
    Menggabungkan seluruh fitur pemain dan kinematika bola dari satu frame.
    """
    frame_id = record["frame_id"]
    ball_detected = record["ball"] is not None

    ball_features = calculate_ball_kinematics(
        record,
        previous_record=previous_record,
        pitch_length=pitch_length,
        pitch_width=pitch_width
    )

    speed_features = calculate_team_speed_features(
        record, team_mapping
    )

    distance_features = {
        "nearest_player_distance": np.nan,
        "nearest_player_team": np.nan,
        "home_min_distance": np.nan,
        "away_min_distance": np.nan,
        "distance_difference": np.nan
    }

    near_ball_features = {}
    for radius in radii:
        near_ball_features[f"home_players_{radius}m"] = np.nan
        near_ball_features[f"away_players_{radius}m"] = np.nan

    if ball_detected:
        distances = calculate_player_ball_distances(
            record,
            pitch_length=pitch_length,
            pitch_width=pitch_width
        )
        nearest = analyze_nearest_player(distances, team_mapping)
        team_distances = calculate_team_ball_distances(distances, team_mapping)

        distance_features["nearest_player_distance"] = (
            nearest["nearest_distance"]
            if nearest["nearest_distance"] is not None else np.nan
        )
        distance_features["nearest_player_team"] = (
            nearest["nearest_team"]
            if nearest["nearest_team"] is not None else np.nan
        )
        distance_features["home_min_distance"] = team_distances["home_min_distance"]
        distance_features["away_min_distance"] = team_distances["away_min_distance"]
        distance_features["distance_difference"] = team_distances["distance_difference"]

        near_counts = count_players_near_ball(distances, team_mapping, radii)
        for radius in radii:
            near_ball_features[f"home_players_{radius}m"] = near_counts[radius]["home_count"]
            near_ball_features[f"away_players_{radius}m"] = near_counts[radius]["away_count"]

    return {
        "frame_id": frame_id,
        "ball_detected": int(ball_detected),
        **ball_features,
        **distance_features,
        **near_ball_features,
        **speed_features
    }


def postprocess_features_df(df, max_gap=50, fps=25):
    """
    Menambal celah pendek bola hilang (<= max_gap frame) dengan interpolasi linier
    dan menghitung sudut belokan arah bola (ball_angle_diff) dalam derajat.
    """
    df = df.copy()

    # Interpolasi posisi bola dan jarak pemain untuk celah hilang pendek (<= 50 frame / 2 detik)
    interp_cols = ["ball_x", "ball_y", "home_min_distance", "away_min_distance"]
    for col in interp_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=max_gap, limit_area="inside")

    # Perbarui selisih jarak dan tim terdekat pada frame hasil interpolasi
    valid_dist = df["home_min_distance"].notna() & df["away_min_distance"].notna()
    df.loc[valid_dist, "distance_difference"] = (
        df.loc[valid_dist, "home_min_distance"] - df.loc[valid_dist, "away_min_distance"]
    ).abs()
    df.loc[valid_dist & df["nearest_player_team"].isna(), "nearest_player_team"] = np.where(
        df.loc[valid_dist & df["nearest_player_team"].isna(), "home_min_distance"]
        <= df.loc[valid_dist & df["nearest_player_team"].isna(), "away_min_distance"],
        0.0,
        1.0
    )

    # Hitung ulang ball_speed pada titik yang diinterpolasi dan sudut belokan vektor
    dx = df["ball_x"].diff()
    dy = df["ball_y"].diff()
    step_dist = np.sqrt(dx**2 + dy**2)
    df["ball_speed"] = df["ball_speed"].fillna(step_dist * fps)

    # Sudut belokan antara vektor (t-1 -> t) dan (t -> t+1)
    dx_next = dx.shift(-1)
    dy_next = dy.shift(-1)
    norm_prev = np.sqrt(dx**2 + dy**2)
    norm_next = np.sqrt(dx_next**2 + dy_next**2)
    denom = norm_prev * norm_next

    cos_angle = np.where(denom > 1e-6, (dx * dx_next + dy * dy_next) / denom, np.nan)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    df["ball_angle_diff"] = np.degrees(np.arccos(cos_angle))

    return df

