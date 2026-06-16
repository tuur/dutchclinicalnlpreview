

import os
import json
import math
import re
import textwrap
import zipfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import pandas as pd


TEXT_UNIT_TERMS = [
    "text",
    "texts",
    "report",
    "reports",
    "note",
    "notes",
    "letter",
    "letters",
    "summary",
    "summaries",
    "record",
    "records",
    "entry",
    "entries",
    "stay",
    "stays",
    "message",
    "messages",
    "fragment",
    "fragments",
    "draft",
    "drafts",
]

PATIENT_UNIT_TERMS = [
    "patient",
    "patients",
    "pt",
    "pts",
    "subject",
    "subjects",
    "participant",
    "participants",
]

PUB_DPI = 300
PUB_SINGLE_COL_WIDTH = 3.35
PUB_DOUBLE_COL_WIDTH = 6.85


def _setup_publication_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.title_fontsize": 8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "savefig.dpi": PUB_DPI,
        "figure.dpi": PUB_DPI,
    })


def process_excel_with_mappings(
    xlsx_path: str,
    columns: list[str],
    output_dir: str
) -> pd.DataFrame:
    """
    Reads an Excel file, creates/updates per-column JSON mapping files,
    and replaces values in the DataFrame using those mappings.

    Parameters
    ----------
    xlsx_path : str
        Path to the Excel file.
    columns : list[str]
        Columns to process.
    output_dir : str
        Directory where JSON files are stored/created.

    Returns
    -------
    pd.DataFrame
        DataFrame with mapped values.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read Excel
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    for col in columns:
        if col not in df.columns:
            continue  # silently skip missing columns

        json_path = os.path.join(output_dir, f"{col}.json")

        # Load existing mapping if present
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        else:
            mapping = {}

        # Extract unique string values in column (excluding NaN)
        values = df[col].dropna().unique()
        string_values = [v for v in values if isinstance(v, str)]

        # Add missing values to mapping (identity mapping)
        updated = False
        for v in string_values:
            if v not in mapping:
                mapping[v] = v
                updated = True

        # Save mapping only if new entries were added or file didn't exist
        if updated or not os.path.exists(json_path):
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)

        # Replace values in DataFrame using mapping (only for strings)
        df[col] = df[col].apply(
            lambda x: mapping.get(x, x) if isinstance(x, str) else x
        )

    return df


def _normalize_number_text(value: str) -> float | None:
    value = str(value).strip()
    if not value:
        return None

    # Treat dots between exactly three trailing digits as thousands separators.
    value = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", value)
    value = value.replace(",", "")

    try:
        return float(value)
    except ValueError:
        return None


def _extract_first_number(value: str) -> float | None:
    match = re.search(r"\d+(?:[,.]\d+)*", str(value))
    if not match:
        return None
    return _normalize_number_text(match.group(0))


def _contains_any(value: str, terms: list[str]) -> bool:
    value = str(value).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", value) for term in terms)


def _infer_sample_unit(ev_size: str, ev_text_type: str, ev_patient_group: str) -> str | None:
    ev_size = str(ev_size).lower()
    ev_text_type = str(ev_text_type).lower()
    ev_patient_group = str(ev_patient_group).lower()

    if _contains_any(ev_size, TEXT_UNIT_TERMS):
        return "texts"
    if _contains_any(ev_size, PATIENT_UNIT_TERMS):
        return "patients"

    if _contains_any(ev_text_type, TEXT_UNIT_TERMS):
        return "texts"
    if _contains_any(ev_patient_group, PATIENT_UNIT_TERMS):
        return "patients"

    return None


def parse_eval_sample_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse evaluation sample sizes from ``Ev size`` into text and patient counts.

    Returns a long dataframe with one row per parsed sample-size count. A source
    row can produce both a text count and a patient count when both are present
    in ``Ev size``.
    """

    required_columns = ["Ev size", "Ev text type", "Ev patient group"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    rows = []
    context_columns = [
        col for col in ["Title", "Model abbreviation", "Usage category"]
        if col in df.columns
    ]

    for index, row in df.iterrows():
        raw_ev_size = row["Ev size"]
        if pd.isna(raw_ev_size):
            continue

        ev_size = str(raw_ev_size).strip()
        if not ev_size or ev_size.lower() in {"nr", "nan", "none", "unclear"}:
            continue

        ev_text_type = "" if pd.isna(row["Ev text type"]) else str(row["Ev text type"])
        ev_patient_group = (
            "" if pd.isna(row["Ev patient group"]) else str(row["Ev patient group"])
        )
        parsed_items = []

        explicit_patterns = [
            ("texts", TEXT_UNIT_TERMS),
            ("patients", PATIENT_UNIT_TERMS),
        ]
        for unit, terms in explicit_patterns:
            unit_pattern = "|".join(re.escape(term) for term in terms)
            for match in re.finditer(
                rf"(\d+(?:[,.]\d+)*)\s*(?:{unit_pattern})\b",
                ev_size,
                flags=re.IGNORECASE,
            ):
                value = _normalize_number_text(match.group(1))
                if value is not None:
                    parsed_items.append((unit, value, "explicit_unit"))

        if not parsed_items:
            value = _extract_first_number(ev_size)
            unit = _infer_sample_unit(ev_size, ev_text_type, ev_patient_group)
            if value is not None and unit is not None:
                parsed_items.append((unit, value, "inferred_unit"))

        for unit, value, parse_source in parsed_items:
            parsed_row = {
                "source row": index,
                "unit": unit,
                "sample size": value,
                "parse source": parse_source,
                "Ev size": ev_size,
                "Ev text type": ev_text_type,
                "Ev patient group": ev_patient_group,
            }
            for col in context_columns:
                parsed_row[col] = row[col]
            rows.append(parsed_row)

    columns = [
        "source row",
        *context_columns,
        "unit",
        "sample size",
        "parse source",
        "Ev size",
        "Ev text type",
        "Ev patient group",
    ]
    return pd.DataFrame(rows, columns=columns)


def _log_hist_bins(values: pd.Series, max_bins: int = 30) -> list[float]:
    values = values[values > 0]
    if values.empty:
        return []

    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        return [min_value * 0.9, max_value * 1.1]

    min_log = math.floor(math.log10(min_value))
    max_log = math.ceil(math.log10(max_value))
    bin_count = min(max_bins, max(6, int((max_log - min_log) * 8)))
    return [
        10 ** (min_log + (max_log - min_log) * i / bin_count)
        for i in range(bin_count + 1)
    ]


def _unique_eval_sample_sizes_by_study(parsed: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce parsed evaluation sample sizes to one count per study and size.

    The same study can appear multiple times in the source spreadsheet because
    several models may share the same evaluation description. This helper
    prevents those duplicate study-level entries from being counted more than
    once.
    """

    if parsed.empty:
        return parsed.copy()

    required = ["Title", "unit", "sample size"]
    missing = [col for col in required if col not in parsed.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}. Available columns: {list(parsed.columns)}"
        )

    dedupe_columns = ["Title", "unit", "sample size"]
    if "Usage category" in parsed.columns:
        dedupe_columns.insert(1, "Usage category")

    study_unique = parsed.drop_duplicates(dedupe_columns).copy()
    return study_unique


def _plot_combined_eval_size_histogram(
    parsed: pd.DataFrame,
    output_path: Path,
    title: str,
    xlabel: str,
) -> None:
    import matplotlib.pyplot as plt

    _setup_publication_style()

    unique_parsed = _unique_eval_sample_sizes_by_study(parsed)
    text_values = unique_parsed.loc[unique_parsed["unit"] == "texts", "sample size"].dropna()
    patient_values = unique_parsed.loc[unique_parsed["unit"] == "patients", "sample size"].dropna()
    combined_values = unique_parsed["sample size"].dropna()

    fig, ax = plt.subplots(figsize=(PUB_DOUBLE_COL_WIDTH, 4.6))
    if combined_values.empty:
        ax.text(0.5, 0.5, "No parsed evaluation sample sizes.", ha="center")
        ax.set_axis_off()
    else:
        bins = _log_hist_bins(combined_values)
        if not text_values.empty:
            ax.hist(
                text_values,
                bins=bins,
                color="#2f6f9f",
                alpha=0.55,
                edgecolor="white",
                label=(
                    f"Texts (studies={text_values.index.nunique()}, "
                    f"median={text_values.median():.0f})"
                ),
            )
        if not patient_values.empty:
            ax.hist(
                patient_values,
                bins=bins,
                color="#d46b35",
                alpha=0.50,
                edgecolor="white",
                label=(
                    f"Patients (studies={patient_values.index.nunique()}, "
                    f"median={patient_values.median():.0f})"
                ),
            )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of studies")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=7.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_eval_sample_size_distributions(
    df: pd.DataFrame,
    output_dir: str = "eval_sample_size_distributions",
) -> pd.DataFrame:
    """
    Plot distributions of evaluation sample sizes split into texts and patients.

    Writes:
      - ``parsed_eval_sample_sizes.csv``
      - ``eval_sample_size_distribution_combined.png``
      - ``eval_sample_size_distribution_combined_<usage>.png``
      - ``eval_sample_size_distribution_texts.png``
      - ``eval_sample_size_distribution_patients.png``

    Returns the parsed long dataframe used for plotting.
    """

    parsed = parse_eval_sample_sizes(df)
    parsed = _unique_eval_sample_sizes_by_study(parsed)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    parsed.to_csv(output_dir_path / "parsed_eval_sample_sizes.csv", index=False)

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(".matplotlib_cache").resolve()),
    )
    import matplotlib.pyplot as plt
    _setup_publication_style()

    _plot_combined_eval_size_histogram(
        parsed,
        output_dir_path / "eval_sample_size_distribution_combined.png",
        "Distribution of evaluation sample sizes",
        "Evaluation sample size (log scale)",
    )

    if "Usage category" in parsed.columns:
        for usage_category, usage_df in parsed.groupby("Usage category", sort=True):
            safe_category = re.sub(r"[^a-z0-9]+", "_", str(usage_category).strip().lower())
            safe_category = safe_category.strip("_") or "unknown"
            _plot_combined_eval_size_histogram(
                usage_df,
                output_dir_path / f"eval_sample_size_distribution_combined_{safe_category}.png",
                f"Distribution of evaluation sample sizes: {usage_category}",
                "Evaluation sample size (log scale)",
            )

        plot_eval_sample_size_panels_by_usage_category(
            parsed,
            output_dir=output_dir,
        )

    for unit in ["texts", "patients"]:
        values = parsed.loc[parsed["unit"] == unit, "sample size"].dropna()
        fig, ax = plt.subplots(figsize=(PUB_DOUBLE_COL_WIDTH, 4.6))

        if values.empty:
            ax.text(0.5, 0.5, f"No parsed {unit} sample sizes.", ha="center")
            ax.set_axis_off()
        else:
            bins = _log_hist_bins(values)
            ax.hist(values, bins=bins, color="#2f6f9f", alpha=0.78, edgecolor="white")
            ax.set_xscale("log")
            ax.set_xlabel(f"Evaluation sample size ({unit}, log scale)")
            ax.set_ylabel("Number of studies")
            ax.set_title(
                f"Distribution of evaluation sample sizes in {unit} "
                f"(studies={values.index.nunique()}, median={values.median():.0f})"
            )
            ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(
            output_dir_path / f"eval_sample_size_distribution_{unit}.png",
            dpi=PUB_DPI,
            bbox_inches="tight",
        )
        plt.close(fig)

    return parsed


def plot_eval_sample_size_panels_by_usage_category(
    parsed: pd.DataFrame,
    output_dir: str = "eval_sample_size_distributions",
    output_name: str = "eval_sample_size_distribution_panels_by_usage_category.png",
) -> Path:
    """
    Plot evaluation sample-size distributions in one panel per usage category.

    Within each panel, texts and patients are overlaid, and each evaluation
    size is counted once per study.
    """

    if "Usage category" not in parsed.columns:
        raise ValueError(
            "Missing expected column: Usage category. "
            f"Available columns: {list(parsed.columns)}"
        )

    unique_parsed = _unique_eval_sample_sizes_by_study(parsed)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / output_name

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(".matplotlib_cache").resolve()),
    )
    import matplotlib.pyplot as plt

    _setup_publication_style()

    usage_categories = sorted(unique_parsed["Usage category"].dropna().unique())
    if not usage_categories:
        fig, ax = plt.subplots(figsize=(PUB_DOUBLE_COL_WIDTH, 3.6))
        ax.text(0.5, 0.5, "No usage categories available.", ha="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    n_panels = len(usage_categories)
    ncols = 2 if n_panels > 1 else 1
    nrows = math.ceil(n_panels / ncols)
    fig_height = max(3.3 * nrows, 3.8)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(PUB_DOUBLE_COL_WIDTH, fig_height),
        squeeze=False,
    )

    axes_list = list(axes.flat)
    for ax in axes_list[n_panels:]:
        ax.axis("off")

    legend_handles = {}
    for ax, usage_category in zip(axes_list, usage_categories):
        usage_df = unique_parsed[unique_parsed["Usage category"] == usage_category]
        text_values = usage_df.loc[usage_df["unit"] == "texts", "sample size"].dropna()
        patient_values = usage_df.loc[usage_df["unit"] == "patients", "sample size"].dropna()
        combined_values = usage_df["sample size"].dropna()

        if combined_values.empty:
            ax.text(0.5, 0.5, "No parsed evaluation sample sizes.", ha="center")
            ax.set_axis_off()
            continue

        bins = _log_hist_bins(combined_values)
        if not text_values.empty:
            bars = ax.hist(
                text_values,
                bins=bins,
                color="#2f6f9f",
                alpha=0.55,
                edgecolor="white",
                label="Texts",
            )
            legend_handles.setdefault("Texts", bars[2][0])
        if not patient_values.empty:
            bars = ax.hist(
                patient_values,
                bins=bins,
                color="#d46b35",
                alpha=0.50,
                edgecolor="white",
                label="Patients",
            )
            legend_handles.setdefault("Patients", bars[2][0])

        ax.set_xscale("log")
        ax.set_title(usage_category)
        ax.set_xlabel("Evaluation sample size (log scale)")
        ax.set_ylabel("Number of studies")
        ax.grid(axis="y", alpha=0.25)

    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="lower center",
        ncol=min(2, max(1, len(legend_handles))),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("Distribution of evaluation sample sizes by usage category", y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _architecture_percentages_by_year(
    df: pd.DataFrame,
    usage_category: str | None = None,
) -> pd.DataFrame:
    required_columns = ["Year", "Title", "Model architecture category"]
    if usage_category is not None:
        required_columns.append("Usage category")

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    subset = df[required_columns].copy()
    subset["Year"] = pd.to_numeric(subset["Year"], errors="coerce")
    subset = subset.dropna(subset=["Year", "Title", "Model architecture category"])
    subset["Year"] = subset["Year"].astype(int)
    subset["study_id"] = subset["Title"].astype(str).str.strip()

    if usage_category is not None:
        subset = subset[subset["Usage category"] == usage_category]

    if subset.empty:
        return pd.DataFrame()

    study_arch = subset.drop_duplicates(
        ["Year", "study_id", "Model architecture category"]
    )
    counts = (
        study_arch.groupby(["Year", "Model architecture category"], as_index=False)
        .agg(count=("study_id", "nunique"))
    )
    totals = (
        subset.drop_duplicates(["Year", "study_id"])
        .groupby("Year", as_index=False)
        .agg(total_studies=("study_id", "nunique"))
    )
    counts = counts.merge(totals, on="Year", how="left")
    counts["percentage"] = counts["count"] / counts["total_studies"] * 100.0
    pivot = (
        counts.pivot(index="Year", columns="Model architecture category", values="percentage")
        .fillna(0.0)
        .sort_index()
    )
    return pivot


def _architecture_study_counts_by_year(
    df: pd.DataFrame,
    usage_category: str | None = None,
) -> pd.DataFrame:
    required_columns = ["Year", "Title", "Model architecture category"]
    if usage_category is not None:
        required_columns.append("Usage category")

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    subset = df[required_columns].copy()
    subset["Year"] = pd.to_numeric(subset["Year"], errors="coerce")
    subset = subset.dropna(subset=["Year", "Title", "Model architecture category"])
    subset["Year"] = subset["Year"].astype(int)
    subset["study_id"] = subset["Title"].astype(str).str.strip()

    if usage_category is not None:
        subset = subset[subset["Usage category"] == usage_category]

    if subset.empty:
        return pd.DataFrame()

    study_arch = subset.drop_duplicates(
        ["Year", "study_id", "Model architecture category"]
    )
    counts = (
        study_arch.groupby(["Year", "Model architecture category"], as_index=False)
        .agg(count=("study_id", "nunique"))
    )
    pivot = (
        counts.pivot(index="Year", columns="Model architecture category", values="count")
        .fillna(0)
        .sort_index()
    )
    return pivot


def _plot_stacked_percentage_bars(
    percentage_df: pd.DataFrame,
    output_path: Path,
    title: str,
    ylabel: str = "Percentage of models",
) -> None:
    import matplotlib.pyplot as plt

    _setup_publication_style()

    fig, ax = plt.subplots(figsize=(PUB_DOUBLE_COL_WIDTH, 4.6))
    if percentage_df.empty:
        ax.text(0.5, 0.5, "No data available for this plot.", ha="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    categories = list(percentage_df.columns)
    years = percentage_df.index.to_list()
    colors = plt.cm.tab20.colors

    bottom = pd.Series(0.0, index=percentage_df.index)
    for i, category in enumerate(categories):
        values = percentage_df[category]
        ax.bar(
            years,
            values,
            bottom=bottom,
            label=category,
            color=colors[i % len(colors)],
            width=0.82,
        )
        bottom = bottom + values

    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.set_title(title)
    ax.set_xticks(years)
    ax.set_xticklabels([str(year) for year in years], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Architecture",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_stacked_count_bars(
    count_df: pd.DataFrame,
    output_path: Path,
    title: str,
    ylabel: str = "Number of studies",
) -> None:
    import matplotlib.pyplot as plt

    _setup_publication_style()

    fig, ax = plt.subplots(figsize=(PUB_DOUBLE_COL_WIDTH, 4.6))
    if count_df.empty:
        ax.text(0.5, 0.5, "No data available for this plot.", ha="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    categories = list(count_df.columns)
    years = count_df.index.to_list()
    colors = plt.cm.tab20.colors

    bottom = pd.Series(0, index=count_df.index, dtype=float)
    for i, category in enumerate(categories):
        values = count_df[category].fillna(0)
        ax.bar(
            years,
            values,
            bottom=bottom,
            label=category,
            color=colors[i % len(colors)],
            width=0.82,
        )
        bottom = bottom + values

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.set_title(title)
    ax.set_xticks(years)
    ax.set_xticklabels([str(year) for year in years], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Architecture",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_architecture_study_count_panels_by_usage_category(
    df: pd.DataFrame,
    output_dir: str = "model_architecture_percentages_by_year",
    output_name: str = "architecture_study_counts_by_year_panels.png",
) -> Path:
    """
    Plot study counts by year in one panel figure per usage category.

    The legend is shared across panels so architecture colors remain stable.
    """

    if "Usage category" not in df.columns:
        raise ValueError(
            "Missing expected column: Usage category. "
            f"Available columns: {list(df.columns)}"
        )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, pd.DataFrame]] = []
    for usage_category in sorted(df["Usage category"].dropna().unique()):
        counts = _architecture_study_counts_by_year(df, usage_category=usage_category)
        panels.append((usage_category, counts))

    output_path = output_dir_path / output_name

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(".matplotlib_cache").resolve()),
    )
    import matplotlib.pyplot as plt

    _setup_publication_style()

    if not panels:
        fig, ax = plt.subplots(figsize=(PUB_DOUBLE_COL_WIDTH, 3.6))
        ax.text(0.5, 0.5, "No usage categories available.", ha="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    categories = sorted(
        {
            architecture
            for _, panel_df in panels
            for architecture in panel_df.columns
        }
    )
    colors = {category: plt.cm.tab20(i % 20) for i, category in enumerate(categories)}

    n_panels = len(panels)
    ncols = 2 if n_panels > 1 else 1
    nrows = math.ceil(n_panels / ncols)
    fig_height = max(3.3 * nrows, 3.8)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(PUB_DOUBLE_COL_WIDTH, fig_height),
        squeeze=False,
    )

    axes_list = list(axes.flat)
    for ax in axes_list[n_panels:]:
        ax.axis("off")

    legend_handles = {}
    for ax, (usage_category, count_df) in zip(axes_list, panels):
        years = count_df.index.to_list()
        bottom = pd.Series(0.0, index=count_df.index)

        if count_df.empty:
            ax.text(0.5, 0.5, "No data available.", ha="center")
            ax.set_axis_off()
            continue

        for category in categories:
            values = count_df[category].fillna(0) if category in count_df.columns else pd.Series(0, index=count_df.index, dtype=float)
            bar = ax.bar(
                years,
                values,
                bottom=bottom,
                color=colors[category],
                width=0.82,
                label=category,
            )
            bottom = bottom + values
            if category not in legend_handles:
                legend_handles[category] = bar[0]

        ax.set_title(usage_category)
        ax.set_xlabel("Year")
        ax.set_ylabel("Studies")
        ax.set_xticks(years)
        ax.set_xticklabels([str(year) for year in years], rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="lower center",
        ncol=min(4, max(1, len(legend_handles))),
        frameon=False,
        title="Architecture",
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("Number of studies using each architecture by year", y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_model_architecture_percentages_by_year(
    df: pd.DataFrame,
    output_dir: str = "model_architecture_percentages_by_year",
) -> dict[str, pd.DataFrame]:
    """
    Plot 100% stacked yearly distributions of model architecture categories.

    Writes study-based percentage plots and matching study-count plots,
    both overall and per usage category.
    Returns a dictionary with the percentage pivot tables used for each plot.
    """

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    results = {}

    overall = _architecture_percentages_by_year(df)
    overall_csv = output_dir_path / "architecture_study_percentages_by_year_overall.csv"
    overall.to_csv(overall_csv)
    _plot_stacked_percentage_bars(
        overall,
        output_dir_path / "architecture_study_percentages_by_year_overall.png",
        "Model architecture usage by year (studies)",
        ylabel="Percentage of studies",
    )
    overall_counts = _architecture_study_counts_by_year(df)
    overall_counts.to_csv(output_dir_path / "architecture_study_counts_by_year_overall.csv")
    _plot_stacked_count_bars(
        overall_counts,
        output_dir_path / "architecture_study_counts_by_year_overall.png",
        "Number of studies using each architecture by year",
        ylabel="Number of studies",
    )
    results["overall"] = overall

    if "Usage category" in df.columns:
        for usage_category in sorted(df["Usage category"].dropna().unique()):
            pivot = _architecture_percentages_by_year(df, usage_category=usage_category)
            counts = _architecture_study_counts_by_year(df, usage_category=usage_category)
            safe_category = re.sub(r"[^a-z0-9]+", "_", str(usage_category).strip().lower())
            safe_category = safe_category.strip("_") or "unknown"
            pivot.to_csv(
                output_dir_path / f"architecture_study_percentages_by_year_{safe_category}.csv"
            )
            _plot_stacked_percentage_bars(
                pivot,
                output_dir_path / f"architecture_study_percentages_by_year_{safe_category}.png",
                f"Model architecture usage by year (studies): {usage_category}",
                ylabel="Percentage of studies",
            )
            counts.to_csv(
                output_dir_path / f"architecture_study_counts_by_year_{safe_category}.csv"
            )
            _plot_stacked_count_bars(
                counts,
                output_dir_path / f"architecture_study_counts_by_year_{safe_category}.png",
                f"Number of studies using each architecture by year: {usage_category}",
                ylabel="Number of studies",
            )
            results[usage_category] = pivot

        plot_architecture_study_count_panels_by_usage_category(
            df,
            output_dir=output_dir,
        )

    return results


def create_comparator_graph(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an edge-list of pairwise model comparisons within comparator blocks.

    Each output row represents one unordered pair of rows from the same
    ``Comparator block`` and contains:
      - model abbreviation 1
      - usage category 1
      - model architecture category 1
      - metric value 1
      - model abbreviation 2
      - usage category 2
      - model architecture category 2
      - metric value 2

    Rows without a comparator block, model abbreviation, or metric value are
    excluded. Source row indices are kept because some blocks can contain the
    same model abbreviation more than once with different metric values.
    """

    required_columns = [
        "Comparator block",
        "Model abbreviation",
        "Usage category",
        "Model architecture category",
        "metric value",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    optional_columns = [
        col for col in ["Main metric", "Ev metrics", "Title"] if col in df.columns
    ]
    working_columns = required_columns + optional_columns

    comparator_rows = (
        df[working_columns]
        .dropna(subset=required_columns)
        .copy()
    )

    edges = []
    for block, block_df in comparator_rows.groupby("Comparator block", sort=False):
        for (idx_1, row_1), (idx_2, row_2) in combinations(block_df.iterrows(), 2):
            edge = {
                "Comparator block": block,
                "model abbreviation 1": row_1["Model abbreviation"],
                "usage category 1": row_1["Usage category"],
                "model architecture category 1": row_1["Model architecture category"],
                "metric value 1": row_1["metric value"],
                "model abbreviation 2": row_2["Model abbreviation"],
                "usage category 2": row_2["Usage category"],
                "model architecture category 2": row_2["Model architecture category"],
                "metric value 2": row_2["metric value"],
                "source row 1": idx_1,
                "source row 2": idx_2,
            }

            for col in optional_columns:
                edge[col] = row_1[col]

            edges.append(edge)

    columns = [
        "Comparator block",
        *optional_columns,
        "model abbreviation 1",
        "usage category 1",
        "model architecture category 1",
        "metric value 1",
        "model abbreviation 2",
        "usage category 2",
        "model architecture category 2",
        "metric value 2",
        "source row 1",
        "source row 2",
    ]
    return pd.DataFrame(edges, columns=columns)




def create_model_architecture_win_graph(
    pairwise_df: pd.DataFrame,
    output_path: str | None = "model_architecture_win_graph.png",
    show: bool = False,
    include_self_edges: bool = False,
    usage_category: str | None = None,
):
    """
    Build and visualize a directed graph of model-architecture wins.

    Nodes are model architecture categories. Only pairwise comparisons where
    both models have the same usage category are considered. If
    ``usage_category`` is supplied, comparisons are additionally restricted to
    that usage category. For every retained comparison, the edge is directed as
    winner -> loser: it starts at the model architecture category with the
    higher metric value and points at the model architecture category with the
    lower metric value. Edge weight is the number of such wins. Ties and
    non-numeric metric values are ignored.

    Returns
    -------
    tuple
        ``(graph, edge_weights_df)`` where ``graph`` is a networkx.DiGraph.
    """

    required_columns = [
        "usage category 1",
        "model architecture category 1",
        "metric value 1",
        "usage category 2",
        "model architecture category 2",
        "metric value 2",
    ]
    missing_columns = [col for col in required_columns if col not in pairwise_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(pairwise_df.columns)}"
        )

    comparisons = pairwise_df[required_columns].copy()
    comparisons["metric value 1"] = pd.to_numeric(
        comparisons["metric value 1"], errors="coerce"
    )
    comparisons["metric value 2"] = pd.to_numeric(
        comparisons["metric value 2"], errors="coerce"
    )
    comparisons = comparisons.dropna(subset=required_columns)
    comparisons = comparisons[
        comparisons["usage category 1"] == comparisons["usage category 2"]
    ].copy()
    if usage_category is not None:
        comparisons = comparisons[
            comparisons["usage category 1"] == usage_category
        ].copy()

    node_rows = comparisons.copy()
    comparisons = comparisons[
        comparisons["metric value 1"] != comparisons["metric value 2"]
    ].copy()

    comparisons["winner"] = comparisons.apply(
        lambda row: row["model architecture category 1"]
        if row["metric value 1"] > row["metric value 2"]
        else row["model architecture category 2"],
        axis=1,
    )
    comparisons["loser"] = comparisons.apply(
        lambda row: row["model architecture category 2"]
        if row["metric value 1"] > row["metric value 2"]
        else row["model architecture category 1"],
        axis=1,
    )

    if not include_self_edges:
        comparisons = comparisons[comparisons["winner"] != comparisons["loser"]]

    edge_weights = (
        comparisons.groupby(["winner", "loser"], as_index=False)
        .size()
        .rename(columns={"size": "weight"})
        .sort_values(["weight", "winner", "loser"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "create_model_architecture_win_graph requires networkx. "
            "Install it with `pip install networkx`."
        ) from exc

    graph = nx.DiGraph()
    categories = (
        set(node_rows["model architecture category 1"].dropna())
        | set(node_rows["model architecture category 2"].dropna())
    )
    edge_records = [
        {"source": row.winner, "target": row.loser, "weight": int(row.weight)}
        for row in edge_weights.itertuples(index=False)
    ]
    ordered_categories = _weighted_circular_node_order(categories, edge_records)
    for category in ordered_categories:
        graph.add_node(category)

    for edge in edge_records:
        graph.add_edge(edge["source"], edge["target"], weight=edge["weight"])

    if output_path or show:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(Path(".matplotlib_cache").resolve()),
        )
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
        _setup_publication_style()

        fig, ax = plt.subplots(figsize=(PUB_DOUBLE_COL_WIDTH, 6.45))
        pos = nx.circular_layout(graph)
        node_sizes = [
            920
            + 80
            * (
                graph.in_degree(node, weight="weight")
                + graph.out_degree(node, weight="weight")
            ) ** 0.5
            for node in graph.nodes
        ]
        weights = [graph[u][v]["weight"] for u, v in graph.edges]
        min_log_weight = min((math.log1p(weight) for weight in weights), default=0)
        max_log_weight = max((math.log1p(weight) for weight in weights), default=0)

        def edge_strength(weight: int) -> float:
            if max_log_weight == min_log_weight:
                return 1.0
            return (math.log1p(weight) - min_log_weight) / (
                max_log_weight - min_log_weight
            )

        def edge_width(weight: int) -> float:
            return 0.5 + 5.0 * edge_strength(weight)

        def edge_alpha(weight: int) -> float:
            return 0.18 + 0.72 * edge_strength(weight)

        label_bbox = {
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
        }

        def readable_rotation(angle: float) -> float:
            if angle > 90:
                return angle - 180
            if angle < -90:
                return angle + 180
            return angle

        def curved_label_geometry(start, end, rad: float, t: float = 0.78):
            sx, sy = start
            tx, ty = end
            dx, dy = tx - sx, ty - sy
            length = math.hypot(dx, dy) or 1.0
            normal = (-dy / length, dx / length)
            control = (
                (sx + tx) / 2 + normal[0] * rad * length,
                (sy + ty) / 2 + normal[1] * rad * length,
            )
            one_minus_t = 1 - t
            label_x = (
                one_minus_t ** 2 * sx
                + 2 * one_minus_t * t * control[0]
                + t ** 2 * tx
            )
            label_y = (
                one_minus_t ** 2 * sy
                + 2 * one_minus_t * t * control[1]
                + t ** 2 * ty
            )
            tangent_x = (
                2 * one_minus_t * (control[0] - sx)
                + 2 * t * (tx - control[0])
            )
            tangent_y = (
                2 * one_minus_t * (control[1] - sy)
                + 2 * t * (ty - control[1])
            )
            angle = math.degrees(math.atan2(tangent_y, tangent_x))
            return label_x, label_y, readable_rotation(angle)

        def trim_edge_to_node_boundaries(source, target, offset: float = 0.16):
            sx, sy = pos[source]
            tx, ty = pos[target]
            dx, dy = tx - sx, ty - sy
            length = math.hypot(dx, dy) or 1.0
            unit_x, unit_y = dx / length, dy / length
            return (
                (sx + unit_x * offset, sy + unit_y * offset),
                (tx - unit_x * offset, ty - unit_y * offset),
            )

        def draw_weighted_edge(source, target, weight: int) -> None:
            alpha = edge_alpha(weight)
            width = edge_width(weight)

            if source == target:
                x, y = pos[source]
                norm = math.hypot(x, y) or 1.0
                radial = (x / norm, y / norm)
                tangent = (-radial[1], radial[0])
                start = (
                    x + radial[0] * 0.17 + tangent[0] * 0.13,
                    y + radial[1] * 0.17 + tangent[1] * 0.13,
                )
                end = (
                    x + radial[0] * 0.17 - tangent[0] * 0.13,
                    y + radial[1] * 0.17 - tangent[1] * 0.13,
                )
                patch = FancyArrowPatch(
                    start,
                    end,
                    connectionstyle="arc3,rad=2.2",
                    arrowstyle="-|>",
                    mutation_scale=10 + 9 * edge_strength(weight),
                    linewidth=width,
                    color="#4a667a",
                    alpha=alpha,
                    zorder=1,
                )
                ax.add_patch(patch)
                label_x = x + radial[0] * 0.44
                label_y = y + radial[1] * 0.44
                label_angle = readable_rotation(
                    math.degrees(math.atan2(tangent[1], tangent[0]))
                )
                ax.text(
                    label_x,
                    label_y,
                    str(weight),
                    fontsize=8,
                    ha="center",
                    va="center",
                    rotation=label_angle,
                    rotation_mode="anchor",
                    bbox=label_bbox,
                    zorder=4,
                )
                return

            rad = 0.22
            start, end = trim_edge_to_node_boundaries(source, target)
            patch = FancyArrowPatch(
                start,
                end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=10 + 9 * edge_strength(weight),
                linewidth=width,
                color="#4a667a",
                alpha=alpha,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
            ax.add_patch(patch)

            label_x, label_y, label_angle = curved_label_geometry(
                start,
                end,
                rad,
            )
            ax.text(
                label_x,
                label_y,
                str(weight),
                fontsize=8,
                ha="center",
                va="center",
                rotation=label_angle,
                rotation_mode="anchor",
                bbox=label_bbox,
                zorder=4,
            )

        for source, target in graph.edges:
            draw_weighted_edge(source, target, graph[source][target]["weight"])

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=node_sizes,
            node_color="#d9ecff",
            edgecolors="#234",
            ax=ax,
        )
        for node in graph.nodes:
            x, y = pos[node]
            ax.text(
                x,
                y,
                "\n".join(_wrap_node_label(node, width=20)),
                fontsize=8.1,
                fontweight="bold",
                ha="center",
                va="center",
                linespacing=0.95,
                zorder=5,
            )
        title = "Model architecture wins over other architectures"
        if usage_category is not None:
            title = f"Model architecture wins over other architectures: {usage_category}"
        ax.set_title(title)
        ax.text(
            0.5,
            0.02,
            "Arrow direction: winner -> loser. Number = count of wins.",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333",
        )
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_xlim(-1.92, 1.92)
        ax.set_ylim(-1.82, 1.82)
        fig.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return graph, edge_weights


def _safe_filename(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def _wrap_node_label(value: str, width: int = 22) -> list[str]:
    text = str(value).strip()
    if not text:
        return [""]
    text = re.sub(r"([/\\-])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    lines = textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return lines or [text]


def _normalize_docx_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text if text and text.lower() != "nan" else ""


def _docx_cell_text(value) -> str:
    text = _normalize_docx_text(value)
    return xml_escape(text)


def _write_minimal_docx_table(
    rows: list[list[str]],
    headers: list[str],
    output_path: Path,
    title: str,
) -> None:
    """
    Write a small landscape DOCX containing a single table.

    This keeps the export dependency-free and is sufficient for publication
    tables that mainly need to be copied into a manuscript.
    """

    def paragraph_xml(text: str, bold: bool = False, size: int = 18) -> str:
        bold_xml = "<w:b/>" if bold else ""
        return (
            "<w:p>"
            "<w:pPr><w:spacing w:after=\"80\"/></w:pPr>"
            "<w:r>"
            f"<w:rPr>{bold_xml}<w:sz w:val=\"{size}\"/></w:rPr>"
            f"<w:t>{xml_escape(text)}</w:t>"
            "</w:r>"
            "</w:p>"
        )

    def cell_xml(text: str, width_twips: int | None = None, bold: bool = False) -> str:
        width_xml = f"<w:tcW w:w=\"{width_twips}\" w:type=\"dxa\"/>" if width_twips else ""
        return (
            "<w:tc>"
            "<w:tcPr>"
            f"{width_xml}"
            "</w:tcPr>"
            f"{paragraph_xml(text, bold=bold, size=18)}"
            "</w:tc>"
        )

    col_widths = [700, 1900, 2500, 5200, 4200]
    table_rows = []
    header_cells = [
        cell_xml(headers[i], width_twips=col_widths[i], bold=True)
        for i in range(len(headers))
    ]
    table_rows.append("<w:tr>" + "".join(header_cells) + "</w:tr>")

    for row in rows:
        cells = []
        for i, value in enumerate(row):
            cells.append(cell_xml(value, width_twips=col_widths[i] if i < len(col_widths) else None))
        table_rows.append("<w:tr>" + "".join(cells) + "</w:tr>")

    table_xml = (
        "<w:tbl>"
        "<w:tblPr>"
        "<w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblLayout w:type=\"fixed\"/>"
        "<w:tblLook w:firstRow=\"1\" w:lastRow=\"0\" w:firstColumn=\"1\" "
        "w:lastColumn=\"0\" w:noHBand=\"0\" w:noVBand=\"1\"/>"
        "</w:tblPr>"
        "<w:tblGrid>"
        + "".join(f"<w:gridCol w:w=\"{w}\"/>" for w in col_widths)
        + "</w:tblGrid>"
        + "".join(table_rows)
        + "</w:tbl>"
    )

    core_props = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:dcterms="http://purl.org/dc/terms/"
    xmlns:dcmitype="http://purl.org/dc/dcmitype/"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{xml_escape(title)}</dc:title>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{datetime.now(timezone.utc).isoformat()}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{datetime.now(timezone.utc).isoformat()}</dcterms:modified>
</cp:coreProperties>
"""

    app_props = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
    xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenAI Codex</Application>
</Properties>
"""

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="R1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="R2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="R3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
    xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
    xmlns:o="urn:schemas-microsoft-com:office:office"
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
    xmlns:v="urn:schemas-microsoft-com:vml"
    xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
    xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
    xmlns:w10="urn:schemas-microsoft-com:office:word"
    xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
    xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
    xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
    xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
    xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
    mc:Ignorable="w14 wp14">
  <w:body>
    <w:p>
      <w:pPr>
        <w:jc w:val="center"/>
      </w:pPr>
      <w:r>
        <w:rPr><w:b/><w:sz w:val="22"/></w:rPr>
        <w:t>{xml_escape(title)}</w:t>
      </w:r>
    </w:p>
    {table_xml}
    <w:sectPr>
      <w:pgSz w:w="15840" w:h="12240" w:orient="landscape"/>
      <w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720" w:header="360" w:footer="360" w:gutter="0"/>
      <w:cols w:space="720"/>
    </w:sectPr>
  </w:body>
</w:document>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("docProps/core.xml", core_props)
        archive.writestr("docProps/app.xml", app_props)
        archive.writestr("word/document.xml", document_xml)


def export_available_models_docx(
    df: pd.DataFrame,
    output_path: str = "available_models_table.docx",
) -> Path:
    """
    Export a DOCX table for rows where the model is shared/available.

    The table columns are:
      Ref., Model, NLP task, Description, Model location
    """

    required_columns = [
        "Model shared",
        "Model abbreviation",
        "Overall NLP task",
        "Model description",
        "Code location(s)",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    subset = df.copy()
    subset["Model shared"] = subset["Model shared"].astype(str).str.strip().str.lower()
    subset = subset[subset["Model shared"].isin({"yes", "partially"})].copy()

    if subset.empty:
        rows = []
    else:
        rows = []
        for ref, (_, row) in enumerate(subset.iterrows(), start=1):
            rows.append([
                str(ref),
                _normalize_docx_text(row["Model abbreviation"]),
                _normalize_docx_text(row["Overall NLP task"]),
                _normalize_docx_text(row["Model description"]),
                _normalize_docx_text(row["Code location(s)"]) or "Not listed",
            ])

    output_path = Path(output_path)
    title = "Models available in the dataset"
    headers = ["Ref.", "Model", "NLP task", "Description", "Model location"]
    _write_minimal_docx_table(rows, headers, output_path, title)
    return output_path


def create_model_catalog_dashboard_html(
    df: pd.DataFrame,
    output_path: str = "model_catalog_dashboard.html",
    compare_dashboard_path: str = "model_architecture_win_graphs_interactive.html",
) -> Path:
    """
    Write an HTML dashboard that lists all models with filters and summary stats.

    The table is row-based, while the summary cards report both row counts and
    study-level evaluation sample-size counts.
    """

    required_columns = [
        "Annotator",
        "First author",
        "Year",
        "Model abbreviation",
        "Usage category",
        "NLP Task description",
        "Model shared",
        "Code location(s)",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    rows = []
    for _, row in df.iterrows():
        year_value = pd.to_numeric(row["Year"], errors="coerce")
        rows.append({
            "author": _normalize_docx_text(row["First author"]),
            "abbreviation": _normalize_docx_text(row["Model abbreviation"]),
            "year": int(year_value) if pd.notna(year_value) else "",
            "usage_category": _normalize_docx_text(row["Usage category"]),
            "nlp_task_description": _normalize_docx_text(row["NLP Task description"]),
            "shared": _normalize_docx_text(row["Model shared"]),
            "model_location": _normalize_docx_text(row["Code location(s)"]) or "Not listed",
            "evaluation_flag": _normalize_docx_text(row.get("Ev conducted yes/no", "")),
        })

    parsed_eval = _unique_eval_sample_sizes_by_study(parse_eval_sample_sizes(df))

    total_rows = len(df)
    total_models = int(df["Model abbreviation"].nunique(dropna=True)) if "Model abbreviation" in df.columns else 0
    evaluation_rows = int(
        df["Ev conducted yes/no"].astype(str).str.strip().str.lower().eq("yes").sum()
    ) if "Ev conducted yes/no" in df.columns else 0
    shared_rows = int(
        df["Model shared"].astype(str).str.strip().str.lower().isin({"yes", "partially"}).sum()
    )
    not_shared_rows = int(
        df["Model shared"].astype(str).str.strip().str.lower().eq("no").sum()
    )
    eval_sample_size_entries = int(len(parsed_eval))

    json_data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    nav_compare = compare_dashboard_path
    nav_catalog = output_path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Model Catalog Dashboard</title>
<style>
  :root {{
    --ink: #17212b;
    --muted: #5d6d7e;
    --line: #d6dde5;
    --blue: #1191FA;
    --blue-dark: #004285;
    --blue-soft: #DCEEFF;
    --accent: #FC6039;
    --bg: #FFFFFF;
  }}
  body {{
    margin: 0;
    font-family: Arial, sans-serif;
    color: var(--ink);
    background: var(--bg);
  }}
  header {{
    padding: 18px 22px;
    background: #fff;
    border-bottom: 1px solid var(--line);
  }}
  h1 {{
    margin: 0 0 6px;
    font-size: 22px;
  }}
  .hint {{
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 10px;
  }}
  .nav {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 8px;
  }}
  .nav a {{
    color: var(--blue-dark);
    text-decoration: none;
    font-weight: 800;
    background: #fff;
    border: 1px solid var(--blue);
    border-radius: 999px;
    padding: 8px 12px;
    box-shadow: 0 1px 0 rgba(0, 0, 0, 0.03);
    transition: background-color .12s ease, color .12s ease, border-color .12s ease, transform .12s ease;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }}
  .nav a:hover {{
    background: var(--blue-soft);
    border-color: var(--blue-dark);
    transform: translateY(-1px);
  }}
  .nav a:active {{
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
    transform: translateY(0);
  }}
  .refbar {{
    margin-top: 10px;
    padding: 8px 10px;
    border-left: 4px solid var(--accent);
    background: #F7FBFF;
    color: var(--muted);
    font-size: 12px;
  }}
  .refbar a {{
    color: var(--blue);
    text-decoration: none;
    font-weight: 700;
  }}
  .stats {{
    display: grid;
    grid-template-columns: repeat(6, minmax(120px, 1fr));
    gap: 10px;
    padding: 14px 14px 0;
  }}
  .stat {{
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px 12px;
  }}
  .stat .label {{
    color: var(--muted);
    font-size: 12px;
  }}
  .stat .value {{
    font-size: 20px;
    font-weight: 800;
    margin-top: 4px;
  }}
  .toolbar {{
    display: grid;
    grid-template-columns: 1.3fr 1fr auto;
    gap: 10px;
    padding: 14px;
    align-items: end;
  }}
  .control {{
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px 12px;
  }}
  .control label {{
    display: block;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 6px;
  }}
  input[type="search"], select {{
    width: 100%;
    box-sizing: border-box;
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 8px 10px;
    font: inherit;
    background: #fff;
  }}
  .checkbox-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    min-height: 38px;
  }}
  .checkbox-row input {{
    width: 16px;
    height: 16px;
  }}
  .table-wrap {{
    margin: 0 14px 14px;
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: auto;
    max-height: 68vh;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  thead th {{
    position: sticky;
    top: 0;
    background: #fff;
    z-index: 1;
    border-bottom: 1px solid var(--line);
    text-align: left;
    padding: 9px 8px;
    white-space: nowrap;
  }}
  tbody td {{
    border-bottom: 1px solid var(--line);
    padding: 8px;
    vertical-align: top;
  }}
  tbody tr:hover {{
    background: #f8fbfe;
  }}
  .muted {{
    color: var(--muted);
    font-size: 12px;
  }}
  .pill {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    background: var(--blue-soft);
    color: var(--blue-dark);
    font-weight: 700;
    font-size: 11px;
  }}
  .empty {{
    padding: 18px;
    color: var(--muted);
  }}
  @media (max-width: 1100px) {{
    .stats {{
      grid-template-columns: repeat(2, minmax(120px, 1fr));
    }}
    .toolbar {{
      grid-template-columns: 1fr;
    }}
  }}
</style>
</head>
<body>
<header>
  <h1>Model Catalog Dashboard</h1>
  <div class="hint">Filter the model list by usage category or shared status. Use the navigation links to switch to the comparison dashboard.</div>
  <div class="refbar">This dashboard corresponds to the Zenodo preprint <a href="https://zenodo.org/records/19461436" target="_blank" rel="noopener">Natural language processing and language models for Dutch clinical text: a systematic review</a> (DOI 10.5281/zenodo.19461436).</div>
  <div class="nav">
    <a href="__CATALOG__">Model catalog</a>
    <a href="__COMPARE__">Model comparison dashboard</a>
  </div>
</header>
<section class="stats">
  <div class="stat"><div class="label">Rows</div><div class="value">__TOTAL_ROWS__</div></div>
  <div class="stat"><div class="label">Models</div><div class="value">__TOTAL_MODELS__</div></div>
  <div class="stat"><div class="label">Evaluations</div><div class="value">__EVALUATION_ROWS__</div></div>
  <div class="stat"><div class="label">Shared</div><div class="value">__SHARED_ROWS__</div></div>
  <div class="stat"><div class="label">Not shared</div><div class="value">__NOT_SHARED_ROWS__</div></div>
  <div class="stat"><div class="label">Eval sample sizes</div><div class="value">__EVAL_SAMPLE_SIZE_ENTRIES__</div></div>
</section>
<section class="toolbar">
  <div class="control">
    <label for="search">Search</label>
    <input id="search" type="search" placeholder="Author, abbreviation, location, usage category..." />
  </div>
  <div class="control">
    <label for="usage">Usage category</label>
    <select id="usage"></select>
  </div>
  <div class="control">
    <label>Shared models</label>
    <div class="checkbox-row">
      <input id="sharedOnly" type="checkbox" />
      <span>Show shared only</span>
    </div>
  </div>
</section>
<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>Author</th>
        <th>Abbreviation</th>
        <th>Year</th>
        <th>Usage category</th>
        <th>NLP task description</th>
        <th>Shared</th>
        <th>Model location</th>
      </tr>
    </thead>
    <tbody id="tableBody"></tbody>
  </table>
  <div id="empty" class="empty" style="display:none;">No rows match the selected filters.</div>
</div>
<script>
const DATA = __JSON_DATA__;
const usageSelect = document.getElementById("usage");
const sharedOnly = document.getElementById("sharedOnly");
const search = document.getElementById("search");
const tableBody = document.getElementById("tableBody");
const empty = document.getElementById("empty");

function esc(value) {{
  return (value ?? "").toString()
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}}

function normalize(value) {{
  return (value ?? "").toString().toLowerCase().trim();
}}

function isShared(value) {{
  const v = normalize(value);
  return v === "yes" || v === "partially";
}}

const usageCategories = [...new Set(DATA.map(row => row.usage_category).filter(Boolean))].sort();
usageSelect.innerHTML = '<option value="">All usage categories</option>' + usageCategories.map(value => `<option value="${{esc(value)}}">${{esc(value)}}</option>`).join("");

function passesFilters(row) {{
  const selectedUsage = usageSelect.value;
  if (selectedUsage && row.usage_category !== selectedUsage) return false;
  if (sharedOnly.checked && !isShared(row.shared)) return false;
  const query = normalize(search.value);
  if (!query) return true;
  return [
    row.author,
    row.abbreviation,
    row.year,
    row.usage_category,
    row.nlp_task_description,
    row.shared,
    row.model_location,
  ].some(value => normalize(value).includes(query));
}}

function render() {{
  const filtered = DATA.filter(passesFilters);
  tableBody.innerHTML = filtered.map(row => `
    <tr>
      <td>${{esc(row.author)}}</td>
      <td>${{esc(row.abbreviation)}}</td>
      <td>${{esc(row.year)}}</td>
      <td>${{esc(row.usage_category)}}</td>
      <td>${{esc(row.nlp_task_description)}}</td>
      <td><span class="pill">${{esc(row.shared || "n/a")}}</span></td>
      <td>${{esc(row.model_location)}}</td>
    </tr>
  `).join("");
  empty.style.display = filtered.length ? "none" : "block";
}

usageSelect.addEventListener("change", render);
sharedOnly.addEventListener("change", render);
search.addEventListener("input", render);
render();
</script>
</body>
</html>
"""

    html = (
    html.replace("{{", "{").replace("}}", "}")
        .replace("__TOTAL_ROWS__", str(total_rows))
        .replace("__TOTAL_MODELS__", str(total_models))
        .replace("__EVALUATION_ROWS__", str(evaluation_rows))
        .replace("__SHARED_ROWS__", str(shared_rows))
        .replace("__NOT_SHARED_ROWS__", str(not_shared_rows))
        .replace("__EVAL_SAMPLE_SIZE_ENTRIES__", str(eval_sample_size_entries))
        .replace("__JSON_DATA__", json_data)
        .replace("__CATALOG__", nav_catalog)
        .replace("__COMPARE__", nav_compare)
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path


def _weighted_circular_node_order(nodes, edges) -> list:
    """
    Order nodes so the strongest connected pair is opposite on the circle.

    This makes the largest arrows pass through the middle of the graph instead
    of running around the outside. Remaining nodes are placed by weighted degree
    in alternating slots around the circle.
    """

    nodes = sorted(str(node) for node in nodes)
    if len(nodes) <= 2:
        return nodes

    pair_weights = {}
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        if source == target:
            continue
        pair = tuple(sorted([source, target]))
        pair_weights[pair] = pair_weights.get(pair, 0) + int(edge["weight"])

    if not pair_weights:
        return nodes

    strongest_pair = max(
        pair_weights.items(),
        key=lambda item: (item[1], item[0][0], item[0][1]),
    )[0]
    weighted_degree = {node: 0 for node in nodes}
    for (source, target), weight in pair_weights.items():
        weighted_degree[source] += weight
        weighted_degree[target] += weight

    order = [None] * len(nodes)
    opposite_index = len(nodes) // 2
    order[0] = strongest_pair[0]
    order[opposite_index] = strongest_pair[1]

    remaining_nodes = [
        node for node in nodes
        if node not in set(strongest_pair)
    ]
    remaining_nodes.sort(
        key=lambda node: (weighted_degree.get(node, 0), node),
        reverse=True,
    )

    slot_offsets = []
    for distance in range(1, len(nodes)):
        left_slot = (opposite_index - distance) % len(nodes)
        right_slot = (opposite_index + distance) % len(nodes)
        for slot in [left_slot, right_slot]:
            if slot not in {0, opposite_index} and slot not in slot_offsets:
                slot_offsets.append(slot)

    for node, slot in zip(remaining_nodes, slot_offsets):
        order[slot] = node

    empty_slots = [idx for idx, node in enumerate(order) if node is None]
    for node, slot in zip(remaining_nodes[len(slot_offsets):], empty_slots):
        order[slot] = node

    return [node for node in order if node is not None]


def _architecture_comparison_counts_by_usage_category(
    pairwise_df: pd.DataFrame,
    include_self_edges: bool = False,
) -> pd.DataFrame:
    """Count non-tie architecture comparisons used for each usage-category graph."""

    required_columns = [
        "usage category 1",
        "model architecture category 1",
        "metric value 1",
        "usage category 2",
        "model architecture category 2",
        "metric value 2",
    ]
    missing_columns = [col for col in required_columns if col not in pairwise_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(pairwise_df.columns)}"
        )

    comparisons = pairwise_df[required_columns].copy()
    comparisons["metric value 1"] = pd.to_numeric(
        comparisons["metric value 1"], errors="coerce"
    )
    comparisons["metric value 2"] = pd.to_numeric(
        comparisons["metric value 2"], errors="coerce"
    )
    comparisons = comparisons.dropna(subset=required_columns)
    comparisons = comparisons[
        comparisons["usage category 1"] == comparisons["usage category 2"]
    ].copy()
    usage_categories = sorted(comparisons["usage category 1"].dropna().unique())
    comparisons = comparisons[
        comparisons["metric value 1"] != comparisons["metric value 2"]
    ].copy()

    if not include_self_edges:
        winner_architecture = comparisons.apply(
            lambda row: row["model architecture category 1"]
            if row["metric value 1"] > row["metric value 2"]
            else row["model architecture category 2"],
            axis=1,
        )
        loser_architecture = comparisons.apply(
            lambda row: row["model architecture category 2"]
            if row["metric value 1"] > row["metric value 2"]
            else row["model architecture category 1"],
            axis=1,
        )
        comparisons = comparisons[winner_architecture != loser_architecture]

    counts = (
        comparisons.groupby("usage category 1", as_index=False)
        .size()
        .rename(columns={"usage category 1": "Usage category", "size": "Comparisons"})
    )
    counts = (
        pd.DataFrame({"Usage category": usage_categories})
        .merge(counts, on="Usage category", how="left")
        .fillna({"Comparisons": 0})
    )
    counts["Comparisons"] = counts["Comparisons"].astype(int)
    counts = counts.sort_values(
        ["Comparisons", "Usage category"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return counts


def create_model_architecture_win_graphs_by_usage_category(
    pairwise_df: pd.DataFrame,
    output_dir: str = "model_architecture_win_graphs_by_usage_category",
    include_self_edges: bool = False,
) -> dict[str, tuple]:
    """
    Write one directed architecture-win graph PNG per usage category.

    Returns a dictionary mapping usage category to ``(graph, edge_weights_df)``.
    """

    required_columns = ["usage category 1", "usage category 2"]
    missing_columns = [col for col in required_columns if col not in pairwise_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(pairwise_df.columns)}"
        )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    same_usage = pairwise_df[
        pairwise_df["usage category 1"] == pairwise_df["usage category 2"]
    ]
    usage_categories = sorted(same_usage["usage category 1"].dropna().unique())

    results = {}
    for category in usage_categories:
        output_path = output_dir_path / f"{_safe_filename(category)}.png"
        results[category] = create_model_architecture_win_graph(
            pairwise_df,
            output_path=str(output_path),
            show=False,
            include_self_edges=include_self_edges,
            usage_category=category,
        )

    return results


def create_interactive_model_architecture_graph_html(
    pairwise_df: pd.DataFrame,
    output_path: str = "model_architecture_win_graphs_interactive.html",
    include_self_edges: bool = False,
) -> Path:
    """
    Write one interactive HTML page with an architecture-win graph per category.

    Clicking an arrow shows the model abbreviations, metric values, comparator
    block, study title, and metric behind that aggregate edge.
    """

    required_columns = [
        "usage category 1",
        "model architecture category 1",
        "model abbreviation 1",
        "metric value 1",
        "usage category 2",
        "model architecture category 2",
        "model abbreviation 2",
        "metric value 2",
    ]
    missing_columns = [col for col in required_columns if col not in pairwise_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(pairwise_df.columns)}"
        )

    comparisons = pairwise_df.copy()
    comparisons["metric value 1"] = pd.to_numeric(
        comparisons["metric value 1"], errors="coerce"
    )
    comparisons["metric value 2"] = pd.to_numeric(
        comparisons["metric value 2"], errors="coerce"
    )
    comparisons = comparisons.dropna(subset=required_columns)
    comparisons = comparisons[
        comparisons["usage category 1"] == comparisons["usage category 2"]
    ].copy()
    same_usage_comparisons = comparisons.copy()
    comparisons = comparisons[
        comparisons["metric value 1"] != comparisons["metric value 2"]
    ].copy()

    comparisons["winner_architecture"] = comparisons.apply(
        lambda row: row["model architecture category 1"]
        if row["metric value 1"] > row["metric value 2"]
        else row["model architecture category 2"],
        axis=1,
    )
    comparisons["loser_architecture"] = comparisons.apply(
        lambda row: row["model architecture category 2"]
        if row["metric value 1"] > row["metric value 2"]
        else row["model architecture category 1"],
        axis=1,
    )
    comparisons["winner_model"] = comparisons.apply(
        lambda row: row["model abbreviation 1"]
        if row["metric value 1"] > row["metric value 2"]
        else row["model abbreviation 2"],
        axis=1,
    )
    comparisons["loser_model"] = comparisons.apply(
        lambda row: row["model abbreviation 2"]
        if row["metric value 1"] > row["metric value 2"]
        else row["model abbreviation 1"],
        axis=1,
    )
    comparisons["winner_metric_value"] = comparisons.apply(
        lambda row: row["metric value 1"]
        if row["metric value 1"] > row["metric value 2"]
        else row["metric value 2"],
        axis=1,
    )
    comparisons["loser_metric_value"] = comparisons.apply(
        lambda row: row["metric value 2"]
        if row["metric value 1"] > row["metric value 2"]
        else row["metric value 1"],
        axis=1,
    )

    if not include_self_edges:
        comparisons = comparisons[
            comparisons["winner_architecture"] != comparisons["loser_architecture"]
        ].copy()

    graph_data = {}
    optional_columns = [
        col for col in ["Comparator block", "Title", "Main metric", "Ev metrics"]
        if col in comparisons.columns
    ]

    for usage_category, category_nodes_df in same_usage_comparisons.groupby(
        "usage category 1",
        sort=True,
    ):
        nodes = sorted(
            set(category_nodes_df["model architecture category 1"].dropna())
            | set(category_nodes_df["model architecture category 2"].dropna())
        )
        category_df = comparisons[
            comparisons["usage category 1"] == usage_category
        ]
        edges = []
        grouped = (
            category_df
            .groupby(["winner_architecture", "loser_architecture"], sort=True)
        )
        for (winner, loser), edge_df in grouped:
            details = []
            for row_dict in edge_df.to_dict(orient="records"):
                detail = {
                    "winner_model": row_dict["winner_model"],
                    "winner_architecture": row_dict["winner_architecture"],
                    "winner_metric_value": row_dict["winner_metric_value"],
                    "loser_model": row_dict["loser_model"],
                    "loser_architecture": row_dict["loser_architecture"],
                    "loser_metric_value": row_dict["loser_metric_value"],
                }
                for col in optional_columns:
                    detail[col] = row_dict.get(col)
                details.append(detail)

            edges.append({
                "source": winner,
                "target": loser,
                "weight": int(len(edge_df)),
                "details": details,
            })

        if not edges:
            continue

        edges = sorted(edges, key=lambda edge: edge["weight"], reverse=True)
        graph_data[usage_category] = {
            "nodes": _weighted_circular_node_order(nodes, edges),
            "edges": edges,
        }

    json_text = json.dumps(graph_data, ensure_ascii=False)
    json_text = json_text.replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Model Architecture Win Graphs</title>
<style>
  :root {{
    --ink: #17212b;
    --muted: #5d6d7e;
    --line: #d6dde5;
    --blue: #1191FA;
    --blue-dark: #004285;
    --blue-soft: #DCEEFF;
    --accent: #FC6039;
    --bg: #FFFFFF;
  }}
  body {{
    margin: 0;
    font-family: Arial, sans-serif;
    color: var(--ink);
    background: var(--bg);
  }}
  header {{
    padding: 18px 22px;
    background: #fff;
    border-bottom: 1px solid var(--line);
  }}
  h1 {{
    margin: 0 0 6px;
    font-size: 22px;
  }}
  .hint {{
    color: var(--muted);
    font-size: 13px;
  }}
  .nav {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 8px;
  }}
  .nav a {{
    color: var(--blue-dark);
    text-decoration: none;
    font-weight: 800;
    background: #fff;
    border: 1px solid var(--blue);
    border-radius: 999px;
    padding: 8px 12px;
    box-shadow: 0 1px 0 rgba(0, 0, 0, 0.03);
    transition: background-color .12s ease, color .12s ease, border-color .12s ease, transform .12s ease;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }}
  .nav a:hover {{
    background: var(--blue-soft);
    border-color: var(--blue-dark);
    transform: translateY(-1px);
  }}
  .nav a:active {{
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
    transform: translateY(0);
  }}
  .refbar {{
    margin-top: 10px;
    padding: 8px 10px;
    border-left: 4px solid var(--accent);
    background: #F7FBFF;
    color: var(--muted);
    font-size: 12px;
  }}
  .refbar a {{
    color: var(--blue);
    text-decoration: none;
    font-weight: 700;
  }}
  .layout {{
    display: grid;
    grid-template-columns: minmax(520px, 1fr) minmax(360px, 520px);
    gap: 14px;
    padding: 14px;
  }}
  .panel {{
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: hidden;
  }}
  .toolbar {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px;
    border-bottom: 1px solid var(--line);
  }}
  select {{
    max-width: 520px;
    padding: 8px 10px;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: #fff;
    color: var(--ink);
  }}
  svg {{
    width: 100%;
    height: 720px;
    display: block;
  }}
  .node circle {{
    fill: var(--blue-soft);
    stroke: var(--blue-dark);
    stroke-width: 1.4;
  }}
  .node text {{
    font-size: 10px;
    font-weight: 700;
    fill: var(--ink);
    paint-order: stroke;
    stroke: white;
    stroke-width: 4px;
    stroke-linejoin: round;
    text-anchor: middle;
    dominant-baseline: middle;
  }}
  .edge path {{
    fill: none;
    stroke: var(--blue-dark);
    cursor: pointer;
  }}
  .edge path.hit-area {{
    stroke: transparent;
    stroke-width: 22px;
    opacity: 0;
    marker-end: none;
    pointer-events: stroke;
  }}
  .edge text {{
    font-size: 11px;
    font-weight: 700;
    fill: #203040;
    paint-order: stroke;
    stroke: white;
    stroke-width: 4px;
    stroke-linejoin: round;
    pointer-events: none;
    transition: font-size .12s ease;
  }}
  .edge:hover path.visible-edge, .edge.active path.visible-edge {{
    stroke: var(--accent);
    marker-end: url(#arrowhead-hover);
  }}
  .edge:hover text, .edge.active text {{
    font-size: 16px;
  }}
  #details {{
    padding: 14px;
    max-height: 770px;
    overflow: auto;
  }}
  .detail-title {{
    font-size: 16px;
    font-weight: 800;
    margin-bottom: 4px;
  }}
  .detail-sub {{
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 12px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  th, td {{
    border-bottom: 1px solid var(--line);
    padding: 7px 6px;
    text-align: left;
    vertical-align: top;
  }}
  th {{
    position: sticky;
    top: 0;
    background: #fff;
    z-index: 1;
  }}
  .empty {{
    color: var(--muted);
    padding: 16px;
  }}
  @media (max-width: 980px) {{
    .layout {{
      grid-template-columns: 1fr;
    }}
    svg {{
      height: 620px;
    }}
  }}
</style>
</head>
<body>
<header>
  <h1>Model Architecture Wins by Usage Category</h1>
  <div class="hint">Arrow direction is winner → loser. Click an arrow to inspect the model comparisons and study titles behind it.</div>
  <div class="refbar">This dashboard corresponds to the Zenodo preprint <a href="https://zenodo.org/records/19461436" target="_blank" rel="noopener">Natural language processing and language models for Dutch clinical text: a systematic review</a> (DOI 10.5281/zenodo.19461436).</div>
  <div class="nav">
    <a href="model_catalog_dashboard.html">Model catalog</a>
    <a href="model_architecture_win_graphs_interactive.html">Model comparison dashboard</a>
  </div>
</header>
<main class="layout">
  <section class="panel">
    <div class="toolbar">
      <label for="category">Usage category</label>
      <select id="category"></select>
    </div>
    <svg id="graph" viewBox="0 0 900 720" role="img" aria-label="Directed model architecture graph"></svg>
  </section>
  <aside class="panel">
    <div id="details" class="empty">Select a usage category and click an arrow.</div>
  </aside>
</main>
<script id="graph-data" type="application/json">{json_text}</script>
<script>
const DATA = JSON.parse(document.getElementById("graph-data").textContent);
const categorySelect = document.getElementById("category");
const svg = document.getElementById("graph");
const details = document.getElementById("details");
const NS = "http://www.w3.org/2000/svg";

function esc(value) {{
  return (value ?? "").toString()
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}}

function numberText(value) {{
  if (value === null || value === undefined || value === "") return "";
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(4).replace(/0+$/, "").replace(/\\.$/, "") : esc(value);
}}

function wrapLabel(value, width = 18) {{
  const text = (value ?? "").toString().trim();
  if (!text) return [""];
  const normalized = text.replace(/([/\\-])/g, " $1 ").replace(/\\s+/g, " ").trim();
  const words = normalized.split(" ");
  const lines = [];
  let current = "";
  for (const word of words) {{
    if (!current) {{
      current = word;
      continue;
    }}
    if ((current + " " + word).length <= width) {{
      current += " " + word;
    }} else {{
      lines.push(current);
      current = word;
    }}
  }}
  if (current) lines.push(current);
  return lines.length ? lines : [normalized];
}}

function makeSvg(tag, attrs = {{}}, text = "") {{
  const el = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  if (text) el.textContent = text;
  return el;
}}

function curvePoint(start, end, rad, t = 0.78) {{
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy) || 1;
  const normal = {{x: -dy / length, y: dx / length}};
  const control = {{
    x: (start.x + end.x) / 2 + normal.x * rad * length,
    y: (start.y + end.y) / 2 + normal.y * rad * length,
  }};
  const omt = 1 - t;
  const x = omt * omt * start.x + 2 * omt * t * control.x + t * t * end.x;
  const y = omt * omt * start.y + 2 * omt * t * control.y + t * t * end.y;
  const tx = 2 * omt * (control.x - start.x) + 2 * t * (end.x - control.x);
  const ty = 2 * omt * (control.y - start.y) + 2 * t * (end.y - control.y);
  let angle = Math.atan2(ty, tx) * 180 / Math.PI;
  if (angle > 90) angle -= 180;
  if (angle < -90) angle += 180;
  return {{x, y, angle}};
}}

function renderDetails(edge) {{
  details.className = "";
  details.innerHTML = `
    <div class="detail-title">${{esc(edge.source)}} wins over ${{esc(edge.target)}}</div>
    <div class="detail-sub">${{edge.weight}} model comparison(s). Arrow points at the loser.</div>
    <table>
      <thead>
        <tr>
          <th>Winning model</th>
          <th>Win value</th>
          <th>Losing model</th>
          <th>Loss value</th>
          <th>Metric</th>
          <th>Comparator block</th>
          <th>Study title</th>
        </tr>
      </thead>
      <tbody>
        ${{edge.details.map(d => `
          <tr>
            <td>${{esc(d.winner_model)}}<br/><small>${{esc(d.winner_architecture)}}</small></td>
            <td>${{numberText(d.winner_metric_value)}}</td>
            <td>${{esc(d.loser_model)}}<br/><small>${{esc(d.loser_architecture)}}</small></td>
            <td>${{numberText(d.loser_metric_value)}}</td>
            <td>${{esc(d["Main metric"] || d["Ev metrics"] || "")}}</td>
            <td>${{esc(d["Comparator block"] || "")}}</td>
            <td>${{esc(d.Title || "")}}</td>
          </tr>
        `).join("")}}
      </tbody>
    </table>
  `;
}}

function renderGraph(category) {{
  svg.innerHTML = "";
  details.className = "empty";
  details.textContent = "Click an arrow to inspect the comparisons behind it.";

  const data = DATA[category] || {{nodes: [], edges: []}};
  const width = 900;
  const height = 720;
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(width, height) * 0.37;
  const nodeRadius = 38;
  const weights = data.edges.map(e => e.weight);
  const minLog = weights.length ? Math.min(...weights.map(w => Math.log1p(w))) : 0;
  const maxLog = weights.length ? Math.max(...weights.map(w => Math.log1p(w))) : 0;
  const strength = (w) => maxLog === minLog ? 1 : (Math.log1p(w) - minLog) / (maxLog - minLog);

  const defs = makeSvg("defs");
  const marker = makeSvg("marker", {{
    id: "arrowhead",
    markerWidth: 10,
    markerHeight: 10,
    refX: 8,
    refY: 3,
    orient: "auto",
    markerUnits: "strokeWidth",
  }});
  marker.appendChild(makeSvg("path", {{d: "M0,0 L0,6 L9,3 z", fill: "#004285"}}));
  defs.appendChild(marker);
  const hoverMarkerVisible = makeSvg("marker", {{
    id: "arrowhead-hover",
    markerWidth: 10,
    markerHeight: 10,
    refX: 8,
    refY: 3,
    orient: "auto",
    markerUnits: "strokeWidth",
  }});
  hoverMarkerVisible.appendChild(makeSvg("path", {{d: "M0,0 L0,6 L9,3 z", fill: "#FC6039"}}));
  defs.appendChild(hoverMarkerVisible);
  const hoverMarker = makeSvg("marker", {{
    id: "arrowhead-hit-area",
    markerWidth: 18,
    markerHeight: 18,
    refX: 14,
    refY: 6,
    orient: "auto",
    markerUnits: "strokeWidth",
  }});
  hoverMarker.appendChild(makeSvg("path", {{
    d: "M0,0 L0,12 L18,6 z",
    fill: "transparent",
    "pointer-events": "visiblePainted",
  }}));
  defs.appendChild(hoverMarker);
  svg.appendChild(defs);

  const positions = new Map();
  data.nodes.forEach((node, i) => {{
    const angle = -Math.PI / 2 + (2 * Math.PI * i / Math.max(1, data.nodes.length));
    positions.set(node, {{
      x: cx + Math.cos(angle) * radius,
      y: cy + Math.sin(angle) * radius,
    }});
  }});

  data.edges.forEach((edge, i) => {{
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) return;
    const s = strength(edge.weight);
    const strokeWidth = 1 + 5 * s;
    const opacity = 0.18 + 0.72 * s;
    const rad = 0.22;
    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const length = Math.hypot(dx, dy) || 1;
    const ux = dx / length;
    const uy = dy / length;
    const start = {{x: source.x + ux * nodeRadius, y: source.y + uy * nodeRadius}};
    const end = {{x: target.x - ux * nodeRadius, y: target.y - uy * nodeRadius}};
    const normal = {{x: -uy, y: ux}};
    const control = {{
      x: (start.x + end.x) / 2 + normal.x * rad * length,
      y: (start.y + end.y) / 2 + normal.y * rad * length,
    }};
    const path = `M ${{start.x}} ${{start.y}} Q ${{control.x}} ${{control.y}} ${{end.x}} ${{end.y}}`;
    const label = curvePoint(start, end, rad);

    const group = makeSvg("g", {{"class": "edge"}});
    group.appendChild(makeSvg("path", {{
      class: "hit-area",
      d: path,
      "marker-end": "url(#arrowhead-hit-area)",
    }}));
    group.appendChild(makeSvg("path", {{
      class: "visible-edge",
      d: path,
      "stroke-width": strokeWidth,
      opacity,
      "marker-end": "url(#arrowhead)",
    }}));
    group.appendChild(makeSvg("text", {{
      x: label.x,
      y: label.y,
      transform: `rotate(${{label.angle}} ${{label.x}} ${{label.y}})`,
      "text-anchor": "middle",
      "dominant-baseline": "central",
    }}, edge.weight.toString()));
    group.addEventListener("click", () => {{
      svg.querySelectorAll(".edge").forEach(el => el.classList.remove("active"));
      group.classList.add("active");
      renderDetails(edge);
    }});
    svg.appendChild(group);
  }});

  data.nodes.forEach(node => {{
    const p = positions.get(node);
    const group = makeSvg("g", {{"class": "node"}});
    group.appendChild(makeSvg("circle", {{cx: p.x, cy: p.y, r: nodeRadius}}));
    const text = makeSvg("text", {{
      x: p.x,
      y: p.y,
    }});
    const lines = wrapLabel(node, 20);
    const startDy = -((lines.length - 1) * 0.55);
    lines.forEach((line, idx) => {{
      const tspan = makeSvg("tspan", {{
        x: p.x,
        dy: idx === 0 ? `${{startDy}}em` : "1.1em",
      }}, line);
      text.appendChild(tspan);
    }});
    group.appendChild(text);
    svg.appendChild(group);
  }});
}}

Object.keys(DATA).forEach(category => {{
  const option = document.createElement("option");
  option.value = category;
  option.textContent = category;
  categorySelect.appendChild(option);
}});

categorySelect.addEventListener("change", () => renderGraph(categorySelect.value));
if (categorySelect.options.length) {{
  categorySelect.selectedIndex = 0;
  renderGraph(categorySelect.value);
}} else {{
  details.textContent = "No cross-architecture comparisons available.";
}}
</script>
</body>
</html>
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def create_usage_category_win_graph(*args, **kwargs):
    """
    Backward-compatible wrapper for create_model_architecture_win_graph.

    The graph is architecture-based; usage categories are used only to restrict
    comparisons to models from the same usage category.
    """
    return create_model_architecture_win_graph(*args, **kwargs)


if __name__ == "__main__":
    file_path = "../Data extraction Dutch cNLP tools 10Jun2026.xlsx"
    df = process_excel_with_mappings(
        file_path,
        ["Dev region", "Ev region", "NLP Task description", "Dev size", "Ev size"],
        "raw_data_mappings",
    )

    pairwise = create_comparator_graph(df)
    graph, edge_weights = create_model_architecture_win_graph(pairwise)
    category_graphs = create_model_architecture_win_graphs_by_usage_category(pairwise)
    interactive_html = create_interactive_model_architecture_graph_html(pairwise)
    comparison_counts = _architecture_comparison_counts_by_usage_category(pairwise)
    parsed_eval_sizes = plot_eval_sample_size_distributions(df)
    architecture_year_plots = plot_model_architecture_percentages_by_year(df)
    model_catalog_html = create_model_catalog_dashboard_html(df)
    available_models_docx = export_available_models_docx(df)
    eval_size_counts = (
        parsed_eval_sizes.groupby("unit", as_index=False)
        .size()
        .rename(columns={"size": "Parsed rows"})
    )

    print(pairwise.head())
    print(edge_weights)
    print("\nNumber of cross-architecture comparisons per usage category:")
    print(comparison_counts.to_string(index=False))
    print("\nParsed evaluation sample-size rows:")
    print(eval_size_counts.to_string(index=False))
    print("\nArchitecture-by-year plots written for:")
    print(", ".join(str(key) for key in architecture_year_plots.keys()))
    print("Wrote matching study-count plots to model_architecture_percentages_by_year/.")
    print("Wrote panel study-count figure to model_architecture_percentages_by_year/architecture_study_counts_by_year_panels.png.")
    print(f"Wrote graph visualization with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(
        "Wrote separate usage-category graph visualizations to "
        "model_architecture_win_graphs_by_usage_category/ "
        f"({len(category_graphs)} files)."
    )
    print(f"Wrote interactive graph HTML to {interactive_html}.")
    print(f"Wrote model catalog HTML to {model_catalog_html}.")
    print("Wrote evaluation sample-size distributions to eval_sample_size_distributions/.")
    print("Wrote architecture-by-year distributions to model_architecture_percentages_by_year/.")
    print(f"Wrote available-model table to {available_models_docx}.")
