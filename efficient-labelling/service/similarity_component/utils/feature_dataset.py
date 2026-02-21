import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    """
    A torch Dataset that loads pre-extracted feature vectors and labels
    from disk. Supports .csv, .parquet, or .npz inputs.

    .csv/.parquet: columns [..., feature_D-1, label]
    .npz (single array): key="data", shape (N, D+1)
    .npz (two arrays): keys="feats","labels"
    """
    def __init__(
        self,
        name: str,
        root: str,
        feature_root: str = "features",
        use_parquet: bool = False,
        use_npz: bool = False,
        feature_dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.int64,
        transform=None,
        target_transform=None,
        max_rows: int = None,  # NEW PARAMETER
    ):
        self.transform = transform
        self.target_transform = target_transform

        folder = os.path.join(root, feature_root, name)
        if use_npz:
            file_path = os.path.join(folder, f"{name}_features.npz")
        else:
            ext = "parquet" if use_parquet else "csv"
            file_path = os.path.join(folder, f"{name}_features.{ext}")

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Feature file not found: {file_path}")

        # ─── Load data ──────────────────────────────────────────────────────────
        if use_npz:
            arr = np.load(file_path)
            if "data" in arr:
                data = arr["data"]
                if max_rows is not None:
                    data = data[:max_rows]
                feats = data[:, :-1].astype(np.float32)
                labels = data[:, -1].astype(np.int64)
            elif {"feats", "labels"}.issubset(arr.keys()):
                feats = arr["feats"].astype(np.float32)
                labels = arr["labels"].astype(np.int64)
                if max_rows is not None:
                    feats = feats[:max_rows]
                    labels = labels[:max_rows]
            else:
                raise KeyError(f"Could not find expected keys in {file_path}: {list(arr.keys())}")
        else:
            # CSV or Parquet path
            if use_parquet:
                df = pd.read_parquet(file_path)
                if max_rows is not None:
                    df = df.head(max_rows)
            else:
                df = pd.read_csv(file_path, nrows=max_rows)

            if "label" not in df.columns:
                raise KeyError(f"'label' column not found in {file_path}")

            labels = df["label"].to_numpy(dtype=np.int64)
            feats  = df.drop(columns=["label"]).to_numpy(dtype=np.float32)

        # ─── Convert to tensors ────────────────────────────────────────────────
        self.features = torch.from_numpy(feats).to(dtype=feature_dtype)
        self.labels   = torch.from_numpy(labels).to(dtype=label_dtype)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        feat  = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            feat = self.transform(feat)
        if self.target_transform:
            label = self.target_transform(label)
        return feat, label
