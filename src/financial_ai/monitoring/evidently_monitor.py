from pathlib import Path
import json

import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_dir: str = "reports/evidently",
):
    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    common_columns = [
        column
        for column in reference_data.columns
        if column in current_data.columns
    ]

    reference = reference_data[
        common_columns
    ].copy()

    current = current_data[
        common_columns
    ].copy()

    report = Report(
        [
            DataDriftPreset(
                columns=common_columns,
            )
        ],
        include_tests=True,
    )

    snapshot = report.run(
        current_data=current,
        reference_data=reference,
    )

    html_path = (
        output_path / "feature_drift_report.html"
    )

    json_path = (
        output_path / "feature_drift_report.json"
    )

    snapshot.save_html(
        str(html_path)
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(snapshot.json())

    return {
        "html": str(html_path),
        "json": str(json_path),
    }


def missing_feature_rate(
    data: pd.DataFrame,
) -> float:

    if data.empty:
        return 0.0

    return float(
        data.isna()
        .mean()
        .mean()
    )