"""
Pipeline prediksi: fitur per frame -> probabilitas -> dekode temporal -> CSV.

Perintah yang meregenerasi file submission (deterministik, tanpa acak):

    python -m src.prediction --team NAMA_TIM

Bila experiments/frame_features.csv belum ada, fitur diekstrak dulu dari ATD
mentah. Lokasi data: environment variable MTAH_DATA_DIR, atau
official-repository/more-than-a-hack-2027/data di akar proyek.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import feature_engineering as fe
from src import possession_model as pm
from src import temporal_analysis as ta
from src.data_loader import iter_tracking, load_metadata, load_smart_tagging

N_FRAMES = 199319
HALVES = [
    (pm.FIRST_HALF_START, pm.FIRST_HALF_END),
    (pm.SECOND_HALF_START, pm.SECOND_HALF_END),
]
VALID_LABELS = {"H", "A", "D", "X"}


@dataclass
class PredictionConfig:
    sigma_d: float = 4.5
    max_gap: int = 50
    temporal: ta.TemporalConfig = field(default_factory=ta.TemporalConfig)
    # Bobot prior Smart Tagging. 0 = dimatikan.
    tagging_team_weight: float = 0.0
    tagging_setpiece_weight: float = 0.0


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_paths(root: Path | None = None, data_dir: str | Path | None = None) -> dict:
    root = Path(root) if root else project_root()
    if data_dir is None:
        env = os.environ.get("MTAH_DATA_DIR")
        data_dir = Path(env) if env else root / "official-repository" / "more-than-a-hack-2027" / "data"
    tactical = Path(data_dir) / "tactical"
    return {
        "atd": tactical / "BAR-ATH_tactical_atd.txt",
        "meta": tactical / "BAR-ATH_tactical_atd.xml",
        "tagging": tactical / "BAR-ATH_tactical_smart_tagging.xml",
        "features": root / "experiments" / "frame_features.csv",
    }


# ---------------------------------------------------------------------------
# Fitur
# ---------------------------------------------------------------------------

def extract_all(atd_path, metadata_path, out_csv, progress_every: int = 25000) -> int:
    """
    Tulis fitur setiap frame dari ATD mentah ke CSV (logika sama dengan sel
    ekstraksi di notebook 03). Mengembalikan jumlah frame yang diproses.
    Dipanggil otomatis oleh main() bila frame_features.csv belum ada.
    """
    team_mapping = load_metadata(metadata_path)["team_mapping"]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    writer = None
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        for record in iter_tracking(atd_path):
            features = fe.extract_frame_features(
                record,
                team_mapping,
                pitch_length=105.3,
                pitch_width=68.0,
            )
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(features.keys()), lineterminator="\n")
                writer.writeheader()
            writer.writerow(features)
            processed += 1
            if progress_every and processed % progress_every == 0:
                print(f"Frame diproses: {processed:,}")

    return processed


def align_features(df: pd.DataFrame, n_frames: int = N_FRAMES) -> pd.DataFrame:
    """
    Pastikan tepat satu baris per video frame 0..n_frames-1 (frame_id = frame + 1).
    Frame yang hilang diisi NaN dengan ball_detected = 0.
    """
    frame = df["frame_id"].astype(int).to_numpy() - 1
    df = df.copy()
    df.index = frame
    df = df[~df.index.duplicated(keep="first")]
    df = df.reindex(np.arange(n_frames))
    df["frame_id"] = np.arange(1, n_frames + 1)
    df["ball_detected"] = df["ball_detected"].fillna(0).astype(int)
    return df.reset_index(drop=True)


def load_features(features_csv, config: PredictionConfig) -> pd.DataFrame:
    raw = pd.read_csv(features_csv)
    aligned = align_features(raw)
    return fe.postprocess_features_df(aligned, max_gap=config.max_gap, fps=ta.FPS)


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------

def build_probs(features_df: pd.DataFrame, config: PredictionConfig, events=None) -> np.ndarray:
    probs = pm.compute_possession_probs(features_df, sigma_d=config.sigma_d)
    if events and (config.tagging_team_weight or config.tagging_setpiece_weight):
        probs = ta.apply_tagging_prior(
            probs,
            events,
            team_weight=config.tagging_team_weight,
            setpiece_weight=config.tagging_setpiece_weight,
        )
    return probs


def build_labels(features_df: pd.DataFrame, config: PredictionConfig, events=None) -> np.ndarray:
    """Label akhir (N,) berisi 'H','A','D','X'. X hanya di frame yang tidak dinilai."""
    probs = build_probs(features_df, config, events)
    labels = ta.decode_halves(probs, HALVES, config.temporal)
    labels[~pm.get_graded_mask(len(labels))] = "X"
    return labels


def validate_labels(labels) -> None:
    labels = np.asarray(labels)
    if labels.shape != (N_FRAMES,):
        raise ValueError(f"Jumlah label {labels.shape}, seharusnya ({N_FRAMES},).")
    bad = set(np.unique(labels)) - VALID_LABELS
    if bad:
        raise ValueError(f"Label tidak valid: {sorted(bad)}")


def write_submission(labels, out_path) -> str:
    """Tulis CSV `frame,label` dan kembalikan MD5-nya (untuk uji reproduksi)."""
    validate_labels(labels)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels}).to_csv(
        out_path, index=False, lineterminator="\n"
    )
    return hashlib.md5(out_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", required=True, help="Nama tim, dipakai untuk nama file <team>.csv")
    parser.add_argument("--out", default=None, help="Default: submissions/<team>.csv")
    parser.add_argument("--data-dir", default=None, help="Folder data (berisi tactical/)")
    parser.add_argument("--features", default=None, help="Default: experiments/frame_features.csv")
    parser.add_argument("--tagging-team-weight", type=float, default=0.0)
    parser.add_argument("--tagging-setpiece-weight", type=float, default=0.0)
    args = parser.parse_args(argv)

    root = project_root()
    paths = default_paths(root, args.data_dir)
    features_csv = Path(args.features) if args.features else paths["features"]
    out_csv = Path(args.out) if args.out else root / "submissions" / f"{args.team}.csv"

    if not features_csv.exists():
        print("frame_features.csv belum ada, mengekstrak dari ATD mentah...")
        extract_all(paths["atd"], paths["meta"], features_csv)

    config = PredictionConfig(
        tagging_team_weight=args.tagging_team_weight,
        tagging_setpiece_weight=args.tagging_setpiece_weight,
    )

    events = None
    if config.tagging_team_weight or config.tagging_setpiece_weight:
        events = load_smart_tagging(paths["tagging"])

    features_df = load_features(features_csv, config)
    labels = build_labels(features_df, config, events)
    digest = write_submission(labels, out_csv)

    print(f"Submission: {out_csv}")
    print(f"MD5       : {digest}")
    for half, (start, end) in enumerate(HALVES, start=1):
        summary = ta.summarize_runs(labels, start, end)
        print(f"Babak {half}: " + ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()