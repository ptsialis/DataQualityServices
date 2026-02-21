
import torch
from torch.utils.data import Dataset

from service.similarity_component.utils.feature_dataset import FeatureDataset


def get_feature_dataset(
    name: str,
    root: str,
    feature_root: str = "features",
    use_parquet: bool = False,
    use_npz: bool = False,
    max_rows: int = None,
    **kwargs
):
    return FeatureDataset(
        name=name,
        root=root,
        feature_root=feature_root,
        use_parquet=use_parquet,
        use_npz=use_npz,
        max_rows=max_rows,
        **kwargs
    )




def features_to_class_tensor(feats: torch.Tensor,
        labels: torch.Tensor,
        C: int = None,
        pad_value: float = 0.) -> torch.Tensor:
    """
    feats: (S, F)   labels: (S,)  where S = total samples
    Returns X: (C, N, F), where
        C = number of unique labels  (or given)
        N = max samples in any class
    Classes with <N samples get padded with pad_value.
    """
    device, dtype = feats.device, feats.dtype
    uniq = torch.unique(labels)
    if C is None:
        C = len(uniq)
    # figure out how many samples each class has
    counts = [(labels == c).sum().item() for c in uniq]
    N = max(counts)

    F = feats.size(1)
    X = torch.full((C, N, F), pad_value, device=device, dtype=dtype)

    for i, c in enumerate(uniq):
        class_feats = feats[labels == c]         # (ni, F)
        ni = class_feats.size(0)
        X[i, :ni] = class_feats                  # fill first ni slots

    return X