"""
Analisis dan penghalusan temporal untuk label possession.

Alur yang disarankan:
    probs (N, 3)  ->  apply_tagging_prior (opsional)
                  ->  viterbi_decode      (status H/A/D yang "lengket")
                  ->  enforce_min_run     (buang run yang terlalu pendek)
                  ->  summarize_runs      (cek jumlah & durasi possession)

Indeks status: 0 = H, 1 = A, 2 = D.

Mengapa dekoder urutan dan bukan majority vote: spellF1 menilai possession
sebagai satu objek. Label yang berkedip H-A-H-A menghasilkan ratusan
possession palsu dan merusak skor, walaupun akurasi per frame tampak bagus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H, A, D = 0, 1, 2
STATE_LABELS = np.array(["H", "A", "D"])
FPS = 25


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalConfig:
    """
    Parameter dekoder. Semua probabilitas transisi adalah per frame (25 fps).

    temperature  : meredam emisi. Skor di possession_model adalah heuristik,
                   bukan probabilitas terkalibrasi, jadi dibuat lebih "landai".
    p_switch_ha  : peluang langsung H->A atau A->H dalam satu frame.
    p_to_dead    : peluang H->D atau A->D (bola menjadi mati).
    p_from_dead  : peluang D->H atau D->A (masing-masing) saat restart.
    min_len_ha   : durasi minimum possession (frame). 40 frame = 1,6 detik,
                   proksi untuk "penguasaan terkendali".
    min_len_d    : durasi minimum henti (frame).

    Pilih nilai berdasarkan prinsip (durasi possession rata-rata ~14 detik
    pada data publik), bukan dengan memaksimalkan skor di 5 menit publik.
    """

    temperature: float = 0.5
    p_switch_ha: float = 1.0 / 750.0
    p_to_dead: float = 1.0 / 900.0
    p_from_dead: float = 1.0 / 400.0
    min_len_ha: int = 40
    min_len_d: int = 50


def transition_log_matrix(cfg: TemporalConfig) -> np.ndarray:
    """Matriks log-transisi 3x3 [dari, ke]."""
    p_ha = cfg.p_switch_ha
    p_hd = cfg.p_to_dead
    p_dx = cfg.p_from_dead

    trans = np.array(
        [
            [1.0 - p_ha - p_hd, p_ha, p_hd],
            [p_ha, 1.0 - p_ha - p_hd, p_hd],
            [p_dx, p_dx, 1.0 - 2.0 * p_dx],
        ],
        dtype=np.float64,
    )
    if np.any(trans <= 0):
        raise ValueError("Probabilitas transisi harus menyisakan peluang tetap > 0.")
    return np.log(trans)


# ---------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------

def viterbi_decode(probs: np.ndarray, config: TemporalConfig | None = None) -> np.ndarray:
    """
    Dekode urutan status paling mungkin (0=H, 1=A, 2=D) dari matriks (N, 3).

    Baris dengan NaN dianggap tidak informatif (emisi seragam).
    Jalankan terpisah untuk setiap babak.
    """
    cfg = config or TemporalConfig()
    probs = np.asarray(probs, dtype=np.float64)
    n = probs.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int8)

    probs = np.where(np.isfinite(probs), probs, 1.0 / 3.0)
    emit = cfg.temperature * np.log(np.clip(probs, 1e-6, 1.0))
    log_trans = transition_log_matrix(cfg)

    score = np.empty((n, 3), dtype=np.float64)
    back = np.zeros((n, 3), dtype=np.int8)
    score[0] = emit[0]
    cols = np.arange(3)

    for t in range(1, n):
        cand = score[t - 1][:, None] + log_trans  # [dari, ke]
        best_from = np.argmax(cand, axis=0)
        back[t] = best_from
        score[t] = cand[best_from, cols] + emit[t]

    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


# ---------------------------------------------------------------------------
# Run-length
# ---------------------------------------------------------------------------

def run_length_encode(seq):
    """Mengembalikan (nilai, awal, panjang) untuk setiap run berurutan."""
    seq = np.asarray(seq)
    if seq.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return seq[:0], empty, empty
    change = np.flatnonzero(seq[1:] != seq[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [seq.size]))
    return seq[starts], starts, ends - starts


def _absorb_run(runs: list, i: int) -> list:
    """Gabungkan run ke-i ke salah satu tetangganya, lalu satukan run kembar."""
    state, length = runs[i]
    left = runs[i - 1] if i > 0 else None
    right = runs[i + 1] if i < len(runs) - 1 else None

    if left is None:
        target = i + 1
    elif right is None:
        target = i - 1
    elif left[0] == right[0]:
        target = i - 1
    elif state == D:
        # Henti pendek di antara dua tim berbeda: ikut tetangga yang lebih panjang.
        target = i - 1 if left[1] >= right[1] else i + 1
    else:
        # Possession pendek: lebih baik jadi possession tetangga daripada jadi D.
        left_poss = left[0] != D
        right_poss = right[0] != D
        if left_poss and not right_poss:
            target = i - 1
        elif right_poss and not left_poss:
            target = i + 1
        else:
            target = i - 1 if left[1] >= right[1] else i + 1

    runs[target][1] += length
    del runs[i]

    merged: list = []
    for run in runs:
        if merged and merged[-1][0] == run[0]:
            merged[-1][1] += run[1]
        else:
            merged.append(run)
    return merged


def enforce_min_run(path, min_len_ha: int = 40, min_len_d: int = 50) -> np.ndarray:
    """
    Hapus run yang lebih pendek dari batas minimum, mulai dari yang terpendek.

    Dipakai SETELAH Viterbi. Pada label mentah yang berkedip ribuan kali
    fungsi ini lambat; tidak dimaksudkan untuk itu.
    """
    path = np.asarray(path)
    values, _, lengths = run_length_encode(path)
    runs = [[int(v), int(l)] for v, l in zip(values, lengths)]

    while len(runs) > 1:
        idx, best = -1, None
        for i, (state, length) in enumerate(runs):
            limit = min_len_d if state == D else min_len_ha
            if length < limit and (best is None or length < best):
                idx, best = i, length
        if idx < 0:
            break
        runs = _absorb_run(runs, idx)

    out = np.repeat(
        np.array([r[0] for r in runs], dtype=np.int8),
        np.array([r[1] for r in runs], dtype=np.int64),
    )
    return out


# ---------------------------------------------------------------------------
# Prior dari Smart Tagging (opsional, default dimatikan)
# ---------------------------------------------------------------------------

def apply_tagging_prior(
    probs: np.ndarray,
    events: list,
    team_weight: float = 0.0,
    setpiece_weight: float = 0.0,
) -> np.ndarray:
    """
    Menggeser probabilitas memakai Smart Tagging sebagai BUKTI lemah.

    - Fase non-set-piece dengan Team = Barcelona/Athletic: kalikan peluang tim
      itu dengan (1 + team_weight).
    - Fase SET PIECES: kalikan peluang D dengan (1 + setpiece_weight).

    Tag bersifat otomatis dan bisa salah; 36 instance bertag N/A (termasuk
    semua throw-in) dilewati. Bobot 0 berarti fungsi ini tidak mengubah apa pun.
    """
    out = np.array(probs, dtype=np.float64, copy=True)
    if team_weight == 0.0 and setpiece_weight == 0.0:
        return out.astype(np.float32)

    n = out.shape[0]
    for event in events:
        start = max(int(event["start_frame"]), 0)
        end = min(int(event["end_frame"]), n - 1)
        if end < start:
            continue
        if event["code"] == "SET PIECES":
            out[start : end + 1, D] *= 1.0 + setpiece_weight
        elif event["team_code"] in (0, 1):
            out[start : end + 1, event["team_code"]] *= 1.0 + team_weight

    sums = np.maximum(out.sum(axis=1, keepdims=True), 1e-9)
    return (out / sums).astype(np.float32)


# ---------------------------------------------------------------------------
# Ringkasan
# ---------------------------------------------------------------------------

def possession_runs(labels) -> list:
    """
    Daftar possession (run maksimal H atau A) sebagai (label, awal, akhir_inklusif).
    D dan X memutus possession: H D H adalah dua possession.
    """
    labels = np.asarray(labels)
    values, starts, lengths = run_length_encode(labels)
    runs = []
    for v, s, l in zip(values, starts, lengths):
        if v in ("H", "A"):
            runs.append((str(v), int(s), int(s + l - 1)))
    return runs


def summarize_runs(labels, start: int | None = None, end: int | None = None, fps: int = FPS) -> dict:
    """
    Ringkasan jumlah frame, jumlah possession, dan durasi untuk label 'H','A','D','X'.
    start/end (inklusif) membatasi jendela, misalnya 5 menit publik.
    """
    labels = np.asarray(labels)
    lo = 0 if start is None else start
    hi = len(labels) - 1 if end is None else end
    window = labels[lo : hi + 1]

    poss = possession_runs(window)
    durations = np.array([(e - s + 1) / fps for _, s, e in poss]) if poss else np.zeros(0)

    values, _, lengths = run_length_encode(window)
    dead_runs = lengths[values == "D"]

    direct_switches = 0
    for (l1, _, e1), (l2, s2, _) in zip(poss[:-1], poss[1:]):
        if l1 != l2 and s2 == e1 + 1:
            direct_switches += 1

    return {
        "frames_H": int(np.sum(window == "H")),
        "frames_A": int(np.sum(window == "A")),
        "frames_D": int(np.sum(window == "D")),
        "frames_X": int(np.sum(window == "X")),
        "n_possessions": len(poss),
        "n_dead_stretches": int(dead_runs.size),
        "n_direct_H_A_switches": direct_switches,
        "mean_possession_s": float(durations.mean()) if durations.size else float("nan"),
        "median_possession_s": float(np.median(durations)) if durations.size else float("nan"),
        "mean_dead_s": float(dead_runs.mean() / fps) if dead_runs.size else float("nan"),
    }


def decode_halves(probs, halves, config: TemporalConfig | None = None, apply_min_run: bool = True):
    """
    Dekode setiap babak secara terpisah. Mengembalikan array label str (N,)
    dengan 'X' di luar babak. halves = [(awal, akhir_inklusif), ...].
    """
    cfg = config or TemporalConfig()
    labels = np.full(len(probs), "X", dtype="<U1")
    for start, end in halves:
        path = viterbi_decode(probs[start : end + 1], cfg)
        if apply_min_run:
            path = enforce_min_run(path, cfg.min_len_ha, cfg.min_len_d)
        labels[start : end + 1] = STATE_LABELS[path]
    return labels