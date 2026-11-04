def check_no_missing(df):
    """Stop with an error if any values are missing."""
    n = df.isna().sum().sum()
    if n > 0:
        raise ValueError(f"{n} missing values found")
    return df


def drop_constant_columns(df):
    """Remove columns that never change, since they carry no information."""
    dropped = [c for c in df.columns if df[c].nunique() <= 1]
    print("Dropping constant columns:", dropped)
    return df.drop(columns=dropped)


def clean(df):
    """Run all cleaning steps in order."""
    df = check_no_missing(df)
    df = drop_constant_columns(df)
    return df


if __name__ == "__main__":
    from load_data import load_raw
    df = clean(load_raw())
    print(df.shape)