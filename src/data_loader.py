
import xml.etree.ElementTree as ET
import numpy as np


def load_metadata(metadata_path):
    """
    Membaca metadata XML dan mengambil informasi tim serta pemain.

    Returns:
        Dictionary berisi nama tim, ID tim, dan mapping track slot.
    """

    root = ET.parse(metadata_path).getroot()

    teams = list(root.iter("Team"))

    if len(teams) != 2:
        raise ValueError("Metadata harus memiliki dua tim.")

    home_id = teams[0].get("id")
    away_id = teams[1].get("id")

    home_name = teams[0].findtext("Name")
    away_name = teams[1].findtext("Name")

    players = list(root.iter("Player"))

    team_mapping = []

    for player in players:
        team_id = player.get("teamId")

        if team_id == home_id:
            team_mapping.append(0)

        elif team_id == away_id:
            team_mapping.append(1)

        else:
            raise ValueError(
                "Ditemukan track slot dengan teamId tidak dikenal."
            )

    return {
        "home_name": home_name,
        "away_name": away_name,
        "home_id": home_id,
        "away_id": away_id,
        "team_mapping": np.array(team_mapping, dtype=np.int8)
    }


def parse_tracking_line(line):
    """
    Mengubah satu baris ATD menjadi dictionary Python.

    Data pemain dan bola yang tidak tersedia akan bernilai None.
    """

    parts = line.strip().split(":")

    if len(parts) != 3:
        return None

    frame_id = int(parts[0])

    players = []

    for group in parts[1].split(";"):
        values = group.split(",")

        if len(values) != 3:
            return None

        if values[0] == "NaN":
            players.append(None)

        else:
            players.append({
                "x": float(values[0]),
                "y": float(values[1]),
                "speed": float(values[2])
            })

    if len(players) != 46:
        return None

    ball_values = parts[2].split(",")

    if len(ball_values) != 3:
        return None

    if ball_values[0] == "NaN":
        ball = None

    else:
        ball = {
            "x": float(ball_values[0]),
            "y": float(ball_values[1]),
            "speed": float(ball_values[2])
        }

    return {
        "frame_id": frame_id,
        "players": players,
        "ball": ball
    }


def iter_tracking(file_path):
    """
    Membaca file ATD satu baris demi satu baris.

    Menggunakan yield agar seluruh data tidak dimuat
    sekaligus ke dalam RAM.
    """

    with open(file_path, "r") as file:

        for line in file:

            record = parse_tracking_line(line)

            if record is not None:
                yield record


def inspect_tracking(file_path):
    """
    Memeriksa kualitas data ATD secara bertahap.
    Tidak menyimpan seluruh data ke memori.
    """

    total_frames = 0
    ball_detected = 0
    ball_missing = 0
    player_counts = {}

    first_frame = None
    last_frame = None
    previous_frame = None
    sequential = True

    for record in iter_tracking(file_path):

        frame_id = record["frame_id"]

        if first_frame is None:
            first_frame = frame_id

        if previous_frame is not None:
            if frame_id != previous_frame + 1:
                sequential = False

        previous_frame = frame_id
        last_frame = frame_id

        total_frames += 1

        if record["ball"] is not None:
            ball_detected += 1
        else:
            ball_missing += 1

        available_players = sum(
            player is not None
            for player in record["players"]
        )

        player_counts[available_players] = (
            player_counts.get(available_players, 0) + 1
        )

    return {
        "total_frames": total_frames,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "sequential": sequential,
        "ball_detected": ball_detected,
        "ball_missing": ball_missing,
        "player_counts": player_counts
    }



def inspect_player_tracking(file_path, team_mapping):
    """
    Memeriksa koordinat, kecepatan, dan deteksi pemain
    berdasarkan mapping tim dari metadata.
    """

    total_frames = 0
    available_players = 0
    missing_players = 0

    home_detected = 0
    away_detected = 0

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    min_speed = float("inf")
    max_speed = float("-inf")

    invalid_x = 0
    invalid_y = 0
    invalid_speed = 0

    outside_x = 0
    outside_y = 0

    for record in iter_tracking(file_path):

        total_frames += 1

        for index, player in enumerate(record["players"]):

            if player is None:
                missing_players += 1
                continue

            available_players += 1

            if team_mapping[index] == 0:
                home_detected += 1
            else:
                away_detected += 1

            x = player["x"]
            y = player["y"]
            speed = player["speed"]

            if np.isfinite(x):
                min_x = min(min_x, x)
                max_x = max(max_x, x)

                if x < 0 or x > 1:
                    outside_x += 1
            else:
                invalid_x += 1

            if np.isfinite(y):
                min_y = min(min_y, y)
                max_y = max(max_y, y)

                if y < 0 or y > 1:
                    outside_y += 1
            else:
                invalid_y += 1

            if np.isfinite(speed):
                min_speed = min(min_speed, speed)
                max_speed = max(max_speed, speed)
            else:
                invalid_speed += 1

    return {
        "total_frames": total_frames,
        "available_players": available_players,
        "missing_players": missing_players,
        "home_detected": home_detected,
        "away_detected": away_detected,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "min_speed": min_speed,
        "max_speed": max_speed,
        "invalid_x": invalid_x,
        "invalid_y": invalid_y,
        "invalid_speed": invalid_speed,
        "outside_x": outside_x,
        "outside_y": outside_y
    }


def sample_player_coordinates(file_path, sample_interval=10):
    """
    Mengambil sampel koordinat dan kecepatan pemain.

    sample_interval=10 berarti mengambil setiap frame ke-10.
    Data asli tidak diubah.
    """

    x_values = []
    y_values = []
    speed_values = []

    for record in iter_tracking(file_path):

        # Mengambil setiap frame ke-10
        if (record["frame_id"] - 1) % sample_interval != 0:
            continue

        for player in record["players"]:

            if player is None:
                continue

            x = player["x"]
            y = player["y"]
            speed = player["speed"]

            if np.isfinite(x):
                x_values.append(x)

            if np.isfinite(y):
                y_values.append(y)

            if np.isfinite(speed):
                speed_values.append(speed)

    return {
        "x": np.array(x_values),
        "y": np.array(y_values),
        "speed": np.array(speed_values)
    }
