import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    df = (
        df
        .sort_values(
            "feature_date"
        )
        .reset_index(drop=True)
    )

    n = len(df)

    train_end = int(
        n * train_ratio
    )

    validation_end = int(
        n
        * (
            train_ratio
            + validation_ratio
        )
    )

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:
    ].copy()

    if (
        not train.empty
        and not validation.empty
    ):

        assert (
            train[
                "feature_date"
            ].max()
            <
            validation[
                "feature_date"
            ].min()
        )

    if (
        not validation.empty
        and not test.empty
    ):

        assert (
            validation[
                "feature_date"
            ].max()
            <
            test[
                "feature_date"
            ].min()
        )

    return (
        train,
        validation,
        test,
    )
    
import pandas as pd


def final_holdout_split(
    df: pd.DataFrame,
    holdout_ratio: float = 0.15,
):

    df = (
        df
        .sort_values(
            "feature_date"
        )
        .reset_index(drop=True)
    )

    split_index = int(
        len(df)
        * (
            1 - holdout_ratio
        )
    )

    development = (
        df.iloc[
            :split_index
        ]
        .copy()
    )

    holdout = (
        df.iloc[
            split_index:
        ]
        .copy()
    )

    assert (
        development[
            "feature_date"
        ].max()
        <
        holdout[
            "feature_date"
        ].min()
    )

    return development, holdout


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 4,
):

    df = (
        df
        .sort_values(
            "feature_date"
        )
        .reset_index(drop=True)
    )

    n = len(df)

    if n < 50:
        raise ValueError(
            "Dataset too small for "
            "walk-forward validation."
        )

    initial_train_size = int(
        n * 0.50
    )

    remaining = (
        n - initial_train_size
    )

    validation_size = max(
        1,
        remaining // n_splits,
    )

    folds = []

    for fold in range(
        n_splits
    ):

        train_end = (
            initial_train_size
            + fold
            * validation_size
        )

        validation_end = min(
            train_end
            + validation_size,
            n,
        )

        if validation_end <= train_end:
            break

        train = (
            df.iloc[
                :train_end
            ]
            .copy()
        )

        validation = (
            df.iloc[
                train_end:
                validation_end
            ]
            .copy()
        )

        if validation.empty:
            break

        assert (
            train[
                "feature_date"
            ].max()
            <
            validation[
                "feature_date"
            ].min()
        )

        folds.append(
            (
                train,
                validation,
            )
        )

    return folds