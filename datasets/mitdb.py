"""MIT-BIH Arrhythmia Database — event detection (1-D segmentation).

Each record is a 2-channel ECG sampled at 360 Hz.  Beat annotations are
converted to per-sample integer class labels using the 5-class AAMI grouping:

    0  background (between beats)
    1  N  — Normal / bundle-branch-block / paced
    2  S  — Supraventricular ectopic
    3  V  — Ventricular ectopic
    4  F  — Fusion
    5  Q  — Unknown / pacemaker artefact

Each annotated R-peak is expanded by ±beat_window samples; samples outside
any window are labelled 0 (background).

Data contract output
--------------------
X_train : List[np.ndarray (T_i, 2)]   training portions  (C == 2)
y_train : List[np.ndarray (T_i,)]     int labels 0–5
X_test  : List[np.ndarray (T_j, 2)]   test portions
y_test  : List[np.ndarray (T_j,)]     int labels 0–5
task    : "event_detection"
metrics : ["segment_f1", "iou_segment"]
"""

import numpy as np
from benchopt import BaseDataset


# AAMI beat-type grouping (MIT-BIH annotation symbol → class index)
BEAT_CLASS = {
    # N group
    "N": 1, "L": 1, "R": 1, "e": 1, "j": 1,
    # S group
    "A": 2, "a": 2, "J": 2, "S": 2,
    # V group
    "V": 3, "E": 3,
    # F group
    "F": 4,
    # Q group
    "P": 5, "f": 5, "u": 5,
}

# All 48 standard MIT-BIH record IDs
MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209",
    "210", "212", "213", "214", "215", "217", "219", "220",
    "221", "222", "223", "228", "230", "231", "232", "233", "234",
]


def _load_record(record_id, data_dir):
    """Load one WFDB record and return (signal, labels) as numpy arrays.

    Parameters
    ----------
    record_id : str  e.g. "100"
    data_dir  : str or Path  local directory holding .hea / .dat / .atr files

    Returns
    -------
    signal : np.ndarray (T, 2)  float32
    labels : np.ndarray (T,)    int32  per-sample class 0–5
    """
    raise NotImplementedError


def _make_label_array(n_samples, ann_samples, ann_symbols, beat_window):
    """Convert beat annotations to a per-sample label array.

    Parameters
    ----------
    n_samples    : int
    ann_samples  : np.ndarray (A,) int   sample indices of each annotation
    ann_symbols  : list of str           annotation symbols (len A)
    beat_window  : int                   half-width of label window in samples

    Returns
    -------
    labels : np.ndarray (n_samples,) int32
    """
    raise NotImplementedError


class Dataset(BaseDataset):
    """MIT-BIH Arrhythmia Database for event detection.

    Parameters
    ----------
    record_ids : list of str or "all"
        Which records to include. Defaults to the full 48-record set.
    debug : bool
        If True, use only the first 2 records and truncate to 5 000 samples.
    train_ratio : float
        Fraction of each record used as training data.
    beat_window : int
        Half-width (in samples) of the label window around each R-peak.
        Default 36 ≈ ±100 ms at 360 Hz (covers the QRS complex).
    """

    name = "MITDB"

    requirements = ["wfdb"]

    parameters = {
        "record_ids": ["all"],
        "debug": [False],
        "train_ratio": [0.7],
        "beat_window": [36],
    }

    def get_data(self):
        from benchmark_utils.download import fetch_mitdb

        data_dir = fetch_mitdb()

        record_ids = MITDB_RECORDS if self.record_ids == "all" else self.record_ids
        if self.debug:
            record_ids = record_ids[:2]

        X_train, y_train, X_test, y_test = [], [], [], []
        for rid in record_ids:
            signal, labels = _load_record(rid, data_dir)

            if self.debug:
                signal = signal[:5000]
                labels = labels[:5000]

            split = max(1, int(len(signal) * self.train_ratio))

            X_train.append(signal[:split])
            y_train.append(labels[:split])
            X_test.append(signal[split:])
            y_test.append(labels[split:])

        return dict(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task="event_detection",
            metrics=["segment_f1", "iou_segment"],
        )
