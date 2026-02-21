import pandas as pd

# If you want to keep the old 'impute_mean_mode' from Functions_Services,
# import and delegate here. Otherwise the minimal logic below:
def impute_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].isna().any():
            if strategy == "mean" and pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].fillna(out[col].mean())
            elif strategy == "mode":
                mode = out[col].mode()
                out[col] = out[col].fillna(mode.iloc[0] if not mode.empty else None)
            else:
                out[col] = out[col].fillna(method="ffill").fillna(method="bfill")
    return out
