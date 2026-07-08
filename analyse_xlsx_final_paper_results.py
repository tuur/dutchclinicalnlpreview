

import os
import json
import math
import re
import textwrap
import zipfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from urllib.parse import urlparse
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
PUB_SAMPLE_PLOT_WIDTH = 6.15

PLOT_DEV_COLOR = "#2f6f9f"
PLOT_EVAL_COLOR = "#d46b35"
PLOT_NEUTRAL_COLOR = "#4f6f8f"
PLOT_INK_COLOR = "#17212b"
PLOT_MUTED_COLOR = "#5d6d7e"
PLOT_LINE_COLOR = "#d6dde5"
PLOT_BG_COLOR = "#f7fbff"
PLOT_PANEL_BG_COLOR = "#f6f7f8"
PLOT_TEXT_DARK = "#1a1a1a"
PLOT_TEXT_LIGHT = "#555555"

ARCHITECTURE_COLOR_ORDER = [
    "Rule-based",
    "Traditional ML",
    "Feature Engineering & Other",
    "Deep Learning (Non-Transformer)",
    "Ensemble",
    "Fine-Tuned/Pre-Trained Transformers",
    "Prompt-Based LLM",
]
ARCHITECTURE_COLOR_ALIASES = {
    "Prompt-Based Large Language Models (LLMs)": "Prompt-Based LLM",
}
ARCHITECTURE_COLOR_SEQUENCE = [
    "#6f6f6f",
    "#8c6d31",
    "#4f8a8b",
    "#7a5195",
    "#b08d57",
    "#2f6f9f",
    "#d46b35",
]
ARCHITECTURE_FALLBACK_COLORS = [
    "#5c82b8",
    "#9c6b5f",
    "#4c8c6b",
    "#8a63a8",
]

CONTEXTUAL_QUALIFIER_SENTIMENT_ORDER = [
    "Insufficient",
    "Mixed or moderate",
    "Feasible",
    "Potential",
    "Good or sufficient",
    "Effective",
    "High or very good",
]

TEXT_TYPE_SPLIT_ALIASES = {
    "medical summaries": "Survey/progress report",
    "progress reports": "Survey/progress report",
    "nursing notes": "Nursing note",
    "treatment plans": "Nursing note",
    "histology reports": "Histology/cytology/autopsy report",
    "cytology reports": "Histology/cytology/autopsy report",
    "autopsy reports": "Histology/cytology/autopsy report",
}

NETHERLANDS_PROVINCES = [
    "Groningen",
    "Friesland",
    "Drenthe",
    "Overijssel",
    "Flevoland",
    "Gelderland",
    "Utrecht",
    "North Holland",
    "South Holland",
    "Zeeland",
    "North Brabant",
    "Limburg",
]

BELGIUM_PROVINCES = [
    "West Flanders",
    "East Flanders",
    "Antwerp",
    "Flemish Brabant",
    "Limburg",
    "Hainaut",
    "Walloon Brabant",
    "Namur",
    "Liège",
    "Luxembourg",
]

REGION_TO_PROVINCE_ALIASES = {
    "north holland": "North Holland",
    "south holland": "South Holland",
    "utrecht": "Utrecht",
    "groningen": "Groningen",
    "gelderland": "Gelderland",
    "north brabant": "North Brabant",
    "overijssel": "Overijssel",
    "friesland": "Friesland",
    "drenthe": "Drenthe",
    "flevoland": "Flevoland",
    "limburg": "Limburg",
    "zeeland": "Zeeland",
    "antwerp": "Antwerp",
    "east flanders": "East Flanders",
    "west flanders": "West Flanders",
    "flemish brabant": "Flemish Brabant",
    "walloon brabant": "Walloon Brabant",
    "hainaut": "Hainaut",
    "namur": "Namur",
    "liège": "Liège",
    "liege": "Liège",
    "luxembourg": "Luxembourg",
}

PRETRAINED_TRANSFORMER_CATEGORY = "Fine-Tuned/Pre-Trained Transformers"
PROMPTED_LLM_CATEGORIES = {
    "Prompt-Based LLM",
    "Prompt-Based Large Language Models (LLMs)",
}
BASE_MODEL_COLUMN = "Base model (for fine-tuned transformers and prompted LLMs)"

_PRETRAINED_BASE_PATTERNS = [
    (r"medroberta\.nl", "MedRoBERTa.nl"),
    (r"medroberta", "MedRoBERTa"),
    (r"robbert", "RobBERT"),
    (r"roberta", "RoBERTa"),
    (r"bertje", "BERTje"),
    (r"hagalbert", "HAGALBERT"),
    (r"longformer", "Longformer"),
    (r"dragon", "DRAGON"),
    (r"bert", "BERT"),
]

_PROMPTED_LLM_BASE_PATTERNS = [
    (r"gpt-4", "GPT-4"),
    (r"gpt-3\.5", "GPT-3.5"),
    (r"gemini", "Gemini"),
    (r"deepseek-r1-14b", "DeepSeek-R1-14B"),
    (r"mistral-nemo-12b", "Mistral-Nemo-12B"),
    (r"llama-3\.3-70b", "Llama-3.3-70B"),
    (r"llama-3\.2-3b", "Llama-3.2-3B"),
    (r"llama-3\.1-8b", "Llama-3.1-8B"),
    (r"llama-3(?:\.0)?", "Llama-3"),
    (r"gemma-2-9b", "Gemma-2-9B"),
    (r"gemma-2-2b", "Gemma-2-2B"),
    (r"phi-4-14b", "Phi-4-14B"),
    (r"qwen-2\.5-14b", "Qwen-2.5-14B"),
]


def _json_safe_scalar(value):
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def _canonical_architecture_label(label: str) -> str:
    label = str(label).strip()
    return ARCHITECTURE_COLOR_ALIASES.get(label, label)


def _architecture_color_map(labels: list[str]) -> dict[str, str]:
    labels = [str(label).strip() for label in labels if str(label).strip()]
    if not labels:
        return {}

    known_map = {
        category: color
        for category, color in zip(ARCHITECTURE_COLOR_ORDER, ARCHITECTURE_COLOR_SEQUENCE, strict=False)
    }
    extra_map: dict[str, str] = {}
    extra_index = 0
    ordered_labels = []
    for label in labels:
        canonical = _canonical_architecture_label(label)
        if canonical not in ordered_labels:
            ordered_labels.append(canonical)

    for canonical in ordered_labels:
        if canonical in known_map or canonical in extra_map:
            continue
        extra_map[canonical] = ARCHITECTURE_FALLBACK_COLORS[
            extra_index % len(ARCHITECTURE_FALLBACK_COLORS)
        ]
        extra_index += 1

    color_map = {**known_map, **extra_map}
    return {
        label: color_map.get(_canonical_architecture_label(label), "#8ba6c6")
        for label in labels
    }


def _setup_publication_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11.0,
        "axes.titlesize": 12.0,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "legend.title_fontsize": 9.5,
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

        mapping_was_normalized = False
        if col == "Ev metrics":
            normalized_mapping = {}
            for raw_value, mapped_value in mapping.items():
                normalized = _normalize_ev_metrics_value(raw_value)
                normalized_mapping[raw_value] = normalized or mapped_value
                if normalized and normalized != mapped_value:
                    mapping_was_normalized = True
            mapping = normalized_mapping

        # Extract unique string values in column (excluding NaN)
        values = df[col].dropna().unique()
        string_values = [v for v in values if isinstance(v, str)]

        # Append only: preserve all existing mappings and add any new raw values.
        updated = False
        for v in string_values:
            if v not in mapping:
                mapping[v] = _normalize_ev_metrics_value(v) if col == "Ev metrics" else v
                updated = True

        if updated or not os.path.exists(json_path) or mapping_was_normalized:
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


def _normalize_ev_metrics_value(value: object) -> str:
    text = _normalize_model_text(value)
    if not text:
        return ""

    normalized = re.sub(r"[–—−]", "-", text.lower())

    categories: list[str] = []

    def add(label: str) -> None:
        if label not in categories:
            categories.append(label)

    if re.search(r"\bmacro\s+precision\b", normalized):
        add("Macro precision")
    if re.search(r"\bmacro\s+recall\b", normalized):
        add("Macro recall")

    matched_f1_variant = False
    if re.search(r"\b(micro[- ]averaged|micro)\s+(f1|f-score|f measure|f-measure)\b", normalized) or "micro-f1" in normalized:
        add("Micro F1 score")
        matched_f1_variant = True
    if re.search(r"\b(macro[- ]averaged|macro)\s+(f1|f-score|f measure|f-measure)\b", normalized) or "macro-f1" in normalized:
        add("Macro F1 score")
        matched_f1_variant = True
    if re.search(r"\b(weighted|weighted-)\s+(f1|f-score|f measure|f-measure)\b", normalized) or "weighted-f1" in normalized:
        add("Weighted F1 score")
        matched_f1_variant = True
    if not matched_f1_variant and re.search(r"\b(f1|f-score|f measure|f-measure)\b", normalized):
        add("F1 score")

    if (
        "macro precision" not in normalized
        and ("precision" in normalized or "positive predictive value" in normalized or re.search(r"\bppv\b", normalized))
    ):
        add("Precision/PPV")
    if (
        "macro recall" not in normalized
        and ("recall" in normalized or "sensitivity" in normalized or re.search(r"\bsens\b", normalized))
    ):
        add("Recall/Sensitivity")
    if "specificity" in normalized or re.search(r"\bspec\b", normalized):
        add("Specificity")
    if re.search(r"\bnpv\b", normalized) or "negative predictive value" in normalized:
        add("NPV")
    if "balanced accuracy" in normalized:
        add("Balanced accuracy")
    if re.search(r"(?<!balanced )\baccuracy\b", normalized):
        add("Accuracy")

    if "auc" in normalized or "auroc" in normalized or "area under the roc" in normalized:
        add("AUROC")
    if "auprc" in normalized or "auprcr" in normalized or "aupcr" in normalized:
        add("AUPRC")

    if "gwet ac1" in normalized:
        add("Gwet AC1")
    if "kappa" in normalized:
        add("Cohen's Kappa")
    if "percentage agreement" in normalized:
        add("Percentage agreement")

    if "bleu" in normalized:
        add("BLEU")
    if "rouge-1" in normalized or "rouge 1" in normalized:
        add("ROUGE-1")
    if "rouge-l" in normalized or "rouge l" in normalized:
        add("ROUGE-L")
    if "rouge" in normalized:
        add("ROUGE")
    if "bertscore" in normalized:
        add("BERTScore")

    if "perplexity" in normalized:
        add("Perplexity")
    if re.search(r"\bmae\b", normalized):
        add("MAE")
    if re.search(r"\bmse\b", normalized):
        add("MSE")
    if re.search(r"\brmse\b", normalized):
        add("RMSE")
    if "rsmape" in normalized:
        add("RSMAPE")

    if "pdqi-9" in normalized:
        add("PDQI-9")
    if "usability" in normalized:
        add("Usability")
    if "completeness" in normalized:
        add("Completeness")
    if "correctness" in normalized:
        add("Correctness")
    if "conciseness" in normalized or "conciceness" in normalized:
        add("Conciseness")
    if "trustworthiness" in normalized:
        add("Trustworthiness")
    if re.search(r"\btrust\b", normalized):
        add("Trust")
    if "overall preference" in normalized:
        add("Overall preference")
    elif re.search(r"\bpreference\b", normalized):
        add("Preference")
    if "coherence" in normalized:
        add("Coherence")
    if "relevance" in normalized:
        add("Relevance")
    if "omissions" in normalized:
        add("Omissions")
    if "trivial facts" in normalized:
        add("Trivial facts")
    if "hallucinations" in normalized:
        add("Hallucinations")
    if "additions" in normalized:
        add("Additions")
    if "date disagreements" in normalized:
        add("Date disagreements")
    if "error in hazard ratio" in normalized:
        add("Error in hazard ratio")
    if "error in median progression-free survival" in normalized:
        add("Error in median progression-free survival")
    if "visual comparison kaplan meier curves" in normalized:
        add("Visual comparison Kaplan Meier curves")
    if "average review time" in normalized:
        add("Average review time")
    if "adoption rate" in normalized:
        add("Adoption rate")
    if "usefulness" in normalized:
        add("Usefulness")
    if "factual correctness" in normalized:
        add("Factual correctness")
    if "well-being" in normalized:
        add("Well-being")
    if "clinical efficiency" in normalized:
        add("Clinical efficiency")
    if any(term in normalized for term in ["runtime", "cpu peak", "gpu peak", "time spent", "words adjusted", "word count drafted vs sent"]):
        add("Runtime/efficiency")
    if "clinical utility" in normalized:
        add("Clinical utility")
    if any(term in normalized for term in ["symmetric similarity", "cosine similarity", "similarity"]):
        add("Similarity")

    if not categories:
        return text
    return ", ".join(categories)


def _normalize_model_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\t", " ").strip()
    return re.sub(r"\s+", " ", text)


def _extract_base_model_label(text: object, patterns: list[tuple[str, str]]) -> str:
    haystack = _normalize_model_text(text).lower()
    if not haystack:
        return ""

    for pattern, label in patterns:
        if re.search(pattern, haystack, flags=re.IGNORECASE):
            return label
    return ""


def _base_model_label_for_row(row: pd.Series) -> str:
    explicit_base_model = _normalize_model_text(
        row.get(BASE_MODEL_COLUMN, row.get("Base model", ""))
    )
    if explicit_base_model:
        return explicit_base_model

    category = _normalize_model_text(row.get("Category", ""))
    type_of_model = _normalize_model_text(row.get("Type of model", ""))
    abbreviation = _normalize_model_text(row.get("Model abbreviation", ""))

    if category in PROMPTED_LLM_CATEGORIES:
        return (
            _extract_base_model_label(type_of_model, _PROMPTED_LLM_BASE_PATTERNS)
            or _extract_base_model_label(abbreviation, _PROMPTED_LLM_BASE_PATTERNS)
        )

    if category == PRETRAINED_TRANSFORMER_CATEGORY:
        return (
            _extract_base_model_label(type_of_model, _PRETRAINED_BASE_PATTERNS)
            or _extract_base_model_label(abbreviation, _PRETRAINED_BASE_PATTERNS)
        )

    return ""


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
    from matplotlib.ticker import MaxNLocator

    _setup_publication_style()

    unique_parsed = _unique_eval_sample_sizes_by_study(parsed)
    text_values = unique_parsed.loc[unique_parsed["unit"] == "texts", "sample size"].dropna()
    patient_values = unique_parsed.loc[unique_parsed["unit"] == "patients", "sample size"].dropna()
    combined_values = unique_parsed["sample size"].dropna()
    text_studies = (
        unique_parsed.loc[unique_parsed["unit"] == "texts", "Title"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .nunique()
    )
    patient_studies = (
        unique_parsed.loc[unique_parsed["unit"] == "patients", "Title"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .nunique()
    )

    fig, ax = plt.subplots(figsize=(PUB_SAMPLE_PLOT_WIDTH, 4.6))
    if combined_values.empty:
        ax.text(0.5, 0.5, "No parsed evaluation sample sizes.", ha="center")
        ax.set_axis_off()
    else:
        bins = _log_hist_bins(combined_values)
        if not text_values.empty:
            ax.hist(
                text_values,
                bins=bins,
                color=PLOT_DEV_COLOR,
                alpha=0.55,
                edgecolor="white",
                label=(
                    f"Texts (studies={text_studies}, "
                    f"median={text_values.median():.0f})"
                ),
            )
        if not patient_values.empty:
            ax.hist(
                patient_values,
                bins=bins,
                color=PLOT_EVAL_COLOR,
                alpha=0.50,
                edgecolor="white",
                label=(
                    f"Patients (studies={patient_studies}, "
                    f"median={patient_values.median():.0f})"
                ),
            )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of studies")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=9.0)

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
      - ``eval_sample_size_distribution_combined.pdf``
      - ``eval_sample_size_distribution_combined_<usage>.pdf``
      - ``eval_sample_size_distribution_texts.pdf``
      - ``eval_sample_size_distribution_patients.pdf``

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
    from matplotlib.ticker import MaxNLocator
    _setup_publication_style()

    _plot_combined_eval_size_histogram(
        parsed,
        output_dir_path / "eval_sample_size_distribution_combined.pdf",
        "Distribution of evaluation sample sizes",
        "Evaluation sample size (log scale)",
    )

    if "Usage category" in parsed.columns:
        for usage_category, usage_df in parsed.groupby("Usage category", sort=True):
            safe_category = re.sub(r"[^a-z0-9]+", "_", str(usage_category).strip().lower())
            safe_category = safe_category.strip("_") or "unknown"
            _plot_combined_eval_size_histogram(
                usage_df,
                output_dir_path / f"eval_sample_size_distribution_combined_{safe_category}.pdf",
                f"Distribution of evaluation sample sizes: {usage_category}",
                "Evaluation sample size (log scale)",
            )

        plot_eval_sample_size_panels_by_usage_category(
            parsed,
            output_dir=output_dir,
        )

    for unit in ["texts", "patients"]:
        values = parsed.loc[parsed["unit"] == unit, "sample size"].dropna()
        study_count = (
            parsed.loc[parsed["unit"] == unit, "Title"]
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )
        fig, ax = plt.subplots(figsize=(PUB_SAMPLE_PLOT_WIDTH, 4.6))

        if values.empty:
            ax.text(0.5, 0.5, f"No parsed {unit} sample sizes.", ha="center")
            ax.set_axis_off()
        else:
            bins = _log_hist_bins(values)
            ax.hist(values, bins=bins, color=PLOT_DEV_COLOR, alpha=0.78, edgecolor="white")
            ax.set_xscale("log")
            ax.set_xlabel(f"Evaluation sample size ({unit}, log scale)")
            ax.set_ylabel("Number of studies")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(
                f"Distribution of evaluation sample sizes in {unit} "
                f"(studies={study_count}, median={values.median():.0f})"
            )
            ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(
            output_dir_path / f"eval_sample_size_distribution_{unit}.pdf",
            dpi=PUB_DPI,
            bbox_inches="tight",
        )
        plt.close(fig)

    return parsed


def plot_eval_sample_size_panels_by_usage_category(
    parsed: pd.DataFrame,
    output_dir: str = "eval_sample_size_distributions",
    output_name: str = "eval_sample_size_distribution_panels_by_usage_category.pdf",
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
    from matplotlib.ticker import MaxNLocator

    _setup_publication_style()

    usage_categories = sorted(unique_parsed["Usage category"].dropna().unique())
    if not usage_categories:
        fig, ax = plt.subplots(figsize=(PUB_SAMPLE_PLOT_WIDTH, 3.6))
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
        figsize=(PUB_SAMPLE_PLOT_WIDTH, fig_height),
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
                color=PLOT_DEV_COLOR,
                alpha=0.55,
                edgecolor="white",
                label="Texts",
            )
            legend_handles.setdefault("Texts", bars[2][0])
        if not patient_values.empty:
            bars = ax.hist(
                patient_values,
                bins=bins,
                color=PLOT_EVAL_COLOR,
                alpha=0.50,
                edgecolor="white",
                label="Patients",
            )
            legend_handles.setdefault("Patients", bars[2][0])

        ax.set_xscale("log")
        ax.set_title(usage_category)
        ax.set_xlabel("Evaluation sample size (log scale)")
        ax.set_ylabel("Number of studies")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
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


def _study_text_type_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    required_columns = ["Title", column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    subset = df[required_columns].copy()
    subset = subset.dropna(subset=["Title", column])
    subset["study_id"] = subset["Title"].astype(str).str.strip()
    subset["text_type"] = subset[column].astype(str).str.strip()
    subset = subset[(subset["study_id"] != "") & (subset["text_type"] != "")]
    if subset.empty:
        return pd.DataFrame(columns=["text_type", "count", "percentage"])

    exploded_rows = []
    for _, row in subset.iterrows():
        study_id = row["study_id"]
        for text_type in str(row["text_type"]).split(","):
            label = text_type.strip()
            if label:
                exploded_rows.append({"study_id": study_id, "text_type": label})

    if not exploded_rows:
        return pd.DataFrame(columns=["text_type", "count", "percentage"])

    exploded = pd.DataFrame(exploded_rows).drop_duplicates(["study_id", "text_type"])
    total_studies = exploded["study_id"].nunique()
    counts = (
        exploded.groupby("text_type", as_index=False)
        .agg(count=("study_id", "nunique"))
        .sort_values(["count", "text_type"], ascending=[False, True])
        .reset_index(drop=True)
    )
    counts["percentage"] = counts["count"] / total_studies * 100.0 if total_studies else 0.0
    counts.attrs["total_studies"] = int(total_studies)
    return counts


def _study_text_type_counts_any(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["Title", "Dev text type", "Ev text type"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    subset = df[required_columns].copy()
    subset = subset.dropna(subset=["Title"])
    subset["study_id"] = subset["Title"].astype(str).str.strip()
    subset = subset[subset["study_id"] != ""]
    if subset.empty:
        return pd.DataFrame(columns=["text_type", "count", "percentage"])

    exploded_rows = []
    for _, row in subset.iterrows():
        study_id = row["study_id"]
        for column in ("Dev text type", "Ev text type"):
            value = row.get(column, "")
            if pd.isna(value):
                continue
            for text_type in str(value).split(","):
                label = text_type.strip()
                if label:
                    exploded_rows.append({"study_id": study_id, "text_type": label})

    if not exploded_rows:
        return pd.DataFrame(columns=["text_type", "count", "percentage"])

    exploded = pd.DataFrame(exploded_rows).drop_duplicates(["study_id", "text_type"])
    total_studies = exploded["study_id"].nunique()
    counts = (
        exploded.groupby("text_type", as_index=False)
        .agg(count=("study_id", "nunique"))
        .sort_values(["count", "text_type"], ascending=[False, True])
        .reset_index(drop=True)
    )
    counts["percentage"] = counts["count"] / total_studies * 100.0 if total_studies else 0.0
    counts.attrs["total_studies"] = int(total_studies)
    return counts


def _plot_text_type_study_count_bars(
    counts_df: pd.DataFrame,
    output_path: Path,
    title: str,
    color: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    _setup_publication_style()

    total_studies = int(counts_df.attrs.get("total_studies", 0))
    if counts_df.empty:
        fig, ax = plt.subplots(figsize=(PUB_DOUBLE_COL_WIDTH, 3.8))
        ax.text(0.5, 0.5, "No data available for this plot.", ha="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    wrapped_labels = [
        textwrap.fill(str(label), width=34, break_long_words=False)
        for label in counts_df["text_type"].tolist()
    ]
    counts = counts_df["count"].astype(float).to_numpy()
    percentages = counts_df["percentage"].astype(float).to_numpy()

    fig_height = max(4.0, 0.34 * len(counts_df) + 1.4)
    fig, ax = plt.subplots(figsize=(PUB_DOUBLE_COL_WIDTH, fig_height))
    y_positions = list(range(len(counts_df)))
    ax.barh(y_positions, counts, color=color, alpha=0.86, edgecolor="white")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(wrapped_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of unique studies")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", alpha=0.25)

    xmax = max(counts) if len(counts) else 0
    ax.set_xlim(0, xmax * 1.18 if xmax else 1)
    for y, count, pct in zip(y_positions, counts, percentages, strict=False):
        ax.text(
            count + (xmax * 0.02 if xmax else 0.05),
            y,
            f"{int(count)} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9.0,
        )

    ax.text(
        1.0,
        -0.08,
        f"Unique studies: {total_studies}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        color=PLOT_TEXT_LIGHT,
    )

    fig.subplots_adjust(
        left=0.16,
        right=0.91,
        bottom=0.20,
        top=0.94,
        hspace=0.72,
    )
    fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_text_type_study_distributions(
    df: pd.DataFrame,
    output_dir: str = "text_type_study_distributions",
) -> dict[str, Path]:
    """
    Plot study counts by text type for development and evaluation.

    Each study is counted once per text-type label, so a study can contribute
    to multiple text types when it uses multiple kinds of text.
    """

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(".matplotlib_cache").resolve()),
    )
    _setup_publication_style()

    outputs: dict[str, Path] = {}
    panels = []
    specs = [
        ("Dev text type", "Development", PLOT_DEV_COLOR),
        ("Ev text type", "Evaluation", PLOT_EVAL_COLOR),
    ]
    for column, label, color in specs:
        counts = _study_text_type_counts(df, column)
        counts.to_csv(output_dir_path / f"{column.replace(' ', '_').lower()}_study_counts.csv", index=False)
        panel_path = output_dir_path / f"{column.replace(' ', '_').lower()}_study_counts.pdf"
        _plot_text_type_study_count_bars(
            counts,
            panel_path,
            f"{label} text types by unique studies",
            color,
        )
        outputs[f"{column}_pdf"] = panel_path
        panels.append((label, counts, color))

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    max_len = max((len(counts) for _, counts, _ in panels), default=0)
    fig_height = max(5.2, 0.34 * max_len + 1.8)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(PUB_DOUBLE_COL_WIDTH, fig_height),
        squeeze=False,
    )
    for ax, (label, counts, color) in zip(axes.flat, panels, strict=False):
        if counts.empty:
            ax.text(0.5, 0.5, f"No {label.lower()} text types available.", ha="center")
            ax.set_axis_off()
            continue
        wrapped_labels = [
            textwrap.fill(str(v), width=34, break_long_words=False)
            for v in counts["text_type"].tolist()
        ]
        values = counts["count"].astype(float).to_numpy()
        percentages = counts["percentage"].astype(float).to_numpy()
        y_positions = list(range(len(counts)))
        ax.barh(y_positions, values, color=color, alpha=0.86, edgecolor="white")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(wrapped_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Number of unique studies")
        ax.set_title(f"{label} [n={int(counts.attrs.get('total_studies', 0))}]")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="x", alpha=0.25)
        xmax = max(values) if len(values) else 0
        ax.set_xlim(0, xmax * 1.18 if xmax else 1)
        for y, count, pct in zip(y_positions, values, percentages, strict=False):
            ax.text(
                count + (xmax * 0.02 if xmax else 0.05),
                y,
                f"{int(count)} ({pct:.1f}%)",
                va="center",
                ha="left",
                fontsize=9.0,
            )

    fig.suptitle("Study counts by text type", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    combined_path = output_dir_path / "text_type_study_counts_by_dev_ev.pdf"
    fig.savefig(combined_path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)
    outputs["combined_pdf"] = combined_path

    any_counts = _study_text_type_counts_any(df)
    any_counts.to_csv(output_dir_path / "text_type_study_counts_any_dev_or_ev.csv", index=False)
    any_path = output_dir_path / "text_type_study_counts_any_dev_or_ev.pdf"
    _plot_text_type_study_count_bars(
        any_counts,
        any_path,
        "Text types used in development or evaluation",
        PLOT_NEUTRAL_COLOR,
    )
    outputs["any_pdf"] = any_path
    outputs["any_csv"] = output_dir_path / "text_type_study_counts_any_dev_or_ev.csv"

    panel_counts = [
        ("Development", _study_text_type_counts(df, "Dev text type"), PLOT_DEV_COLOR),
        ("Evaluation", _study_text_type_counts(df, "Ev text type"), PLOT_EVAL_COLOR),
        ("Any dev or eval", any_counts, PLOT_NEUTRAL_COLOR),
    ]
    max_len = max((len(counts) for _, counts, _ in panel_counts), default=0)
    panel_fig_height = max(5.4, 0.34 * max_len + 1.8)
    panel_fig, panel_axes = plt.subplots(
        1,
        len(panel_counts),
        figsize=(PUB_DOUBLE_COL_WIDTH * 1.52, panel_fig_height),
        squeeze=False,
    )
    for ax, (label, counts, color) in zip(panel_axes.flat, panel_counts, strict=False):
        if counts.empty:
            ax.text(0.5, 0.5, f"No {label.lower()} text types available.", ha="center")
            ax.set_axis_off()
            continue
        wrapped_labels = [
            textwrap.fill(str(v), width=34, break_long_words=False)
            for v in counts["text_type"].tolist()
        ]
        values = counts["count"].astype(float).to_numpy()
        percentages = counts["percentage"].astype(float).to_numpy()
        y_positions = list(range(len(counts)))
        ax.barh(y_positions, values, color=color, alpha=0.86, edgecolor="white")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(wrapped_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Number of unique studies")
        ax.set_title(f"{label} [n={int(counts.attrs.get('total_studies', 0))}]")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="x", alpha=0.25)
        xmax = max(values) if len(values) else 0
        ax.set_xlim(0, xmax * 1.18 if xmax else 1)
        for y, count, pct in zip(y_positions, values, percentages, strict=False):
            ax.text(
                count + (xmax * 0.02 if xmax else 0.05),
                y,
                f"{int(count)} ({pct:.1f}%)",
                va="center",
                ha="left",
                fontsize=9.0,
            )
    panel_fig.suptitle("Study counts by text type", y=0.995)
    panel_fig.tight_layout(rect=(0, 0, 1, 0.98))
    panel_path = output_dir_path / "text_type_study_counts_panel.pdf"
    panel_fig.savefig(panel_path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(panel_fig)
    outputs["panel_pdf"] = panel_path

    return outputs


def plot_any_dev_eval_text_region_panel(
    df: pd.DataFrame,
    output_dir: str = "text_type_study_distributions",
) -> Path:
    """
    Plot the combined "any dev or eval" text-type distribution next to the
    combined "any dev or eval" province map.
    """

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(".matplotlib_cache").resolve()),
    )
    import cartopy
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    cartopy.config["data_dir"] = str(Path(".cartopy_cache").resolve())
    Path(cartopy.config["data_dir"]).mkdir(parents=True, exist_ok=True)

    _setup_publication_style()

    any_counts = _study_text_type_counts_any(df)
    province_counts = _study_province_counts_any(df)
    province_cmap = LinearSegmentedColormap.from_list(
        "province_blue_tint",
        ["#f4f8fc", "#d8e7f2", "#b1cde2", "#7fa8ca", PLOT_DEV_COLOR],
    )
    vmax = int(province_counts["count"].max() if not province_counts.empty else 0)

    fig = plt.figure(figsize=(PUB_DOUBLE_COL_WIDTH * 1.75, 6.2))
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 1.35],
        wspace=0.10,
    )
    ax_text = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())

    if any_counts.empty:
        ax_text.text(0.5, 0.5, "No text types available.", ha="center")
        ax_text.set_axis_off()
    else:
        wrapped_labels = [
            textwrap.fill(str(v), width=34, break_long_words=False)
            for v in any_counts["text_type"].tolist()
        ]
        values = any_counts["count"].astype(float).to_numpy()
        percentages = any_counts["percentage"].astype(float).to_numpy()
        y_positions = list(range(len(any_counts)))
        ax_text.barh(
            y_positions,
            values,
            color=PLOT_NEUTRAL_COLOR,
            alpha=0.86,
            edgecolor="white",
        )
        ax_text.set_yticks(y_positions)
        ax_text.set_yticklabels(wrapped_labels)
        ax_text.invert_yaxis()
        ax_text.set_xlabel("Number of unique studies")
        ax_text.set_title(
            f"Text types used in development or evaluation [n={int(any_counts.attrs.get('total_studies', 0))}]"
        )
        ax_text.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_text.grid(axis="x", alpha=0.25)
        xmax = max(values) if len(values) else 0
        ax_text.set_xlim(0, xmax * 1.18 if xmax else 1)
        for y, count, pct in zip(y_positions, values, percentages, strict=False):
            ax_text.text(
                count + (xmax * 0.02 if xmax else 0.05),
                y,
                f"{int(count)} ({pct:.1f}%)",
                va="center",
                ha="left",
                fontsize=9.0,
            )

    ax_map.set_facecolor(PLOT_PANEL_BG_COLOR)
    _plot_combined_province_map(
        ax_map,
        province_counts,
        f"Study counts by province [n={province_counts.attrs.get('total_studies', 0)}]",
        province_cmap,
        vmax,
    )

    sm = plt.cm.ScalarMappable(cmap=province_cmap, norm=Normalize(vmin=0, vmax=max(vmax, 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_text, ax_map], fraction=0.03, pad=0.02)
    cbar.set_label("Number of unique studies")
    fig.suptitle(
        "Studies using text types and regions in either development or evaluation\n"
        "Comma-separated labels are split before counting; nationwide/unassigned region labels are excluded from province allocation.",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = output_dir_path / "any_dev_or_eval_text_region_panel.pdf"
    fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _study_metric_counts_by_usage_category(
    df: pd.DataFrame,
    metric_column: str = "Ev metrics",
) -> pd.DataFrame:
    required_columns = ["Title", "Usage category", metric_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    subset = df[required_columns].copy()
    subset = subset.dropna(subset=["Title", "Usage category", metric_column])
    subset["study_id"] = subset["Title"].astype(str).str.strip()
    subset["usage_category"] = subset["Usage category"].astype(str).str.strip()
    subset["metric"] = subset[metric_column].astype(str).str.strip()
    subset = subset[
        (subset["study_id"] != "")
        & (subset["usage_category"] != "")
        & (subset["metric"] != "")
    ]
    if subset.empty:
        return pd.DataFrame(columns=["Usage category", "metric", "count", "percentage"])

    exploded_rows = []
    for _, row in subset.iterrows():
        study_id = row["study_id"]
        usage_category = row["usage_category"]
        for metric in str(row["metric"]).split(","):
            label = metric.strip()
            if label:
                exploded_rows.append(
                    {
                        "study_id": study_id,
                        "Usage category": usage_category,
                        "metric": label,
                    }
                )

    if not exploded_rows:
        return pd.DataFrame(columns=["Usage category", "metric", "count", "percentage"])

    exploded = pd.DataFrame(exploded_rows).drop_duplicates(
        ["study_id", "Usage category", "metric"]
    )
    usage_totals = (
        exploded.groupby("Usage category", as_index=False)
        .agg(total_studies=("study_id", "nunique"))
    )
    counts = (
        exploded.groupby(["Usage category", "metric"], as_index=False)
        .agg(count=("study_id", "nunique"))
        .merge(usage_totals, on="Usage category", how="left")
        .sort_values(["Usage category", "count", "metric"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    counts["percentage"] = counts["count"] / counts["total_studies"] * 100.0
    counts.attrs["total_studies"] = int(exploded["study_id"].nunique())
    return counts


def plot_ev_metric_overview_by_usage_category(
    df: pd.DataFrame,
    output_dir: str = "ev_metric_overview",
    top_n_metrics: int = 16,
) -> dict[str, Path]:
    """
    Plot an overview of evaluation metrics by usage category.

    The plot uses the normalized ``Ev metrics`` values, splits comma-separated
    multi-metric cells, and counts each study once per metric/usage category.
    """

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(".matplotlib_cache").resolve()),
    )
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    _setup_publication_style()

    counts = _study_metric_counts_by_usage_category(df, "Ev metrics")
    counts.to_csv(output_dir_path / "ev_metrics_by_usage_category_long.csv", index=False)

    if counts.empty:
        fig, ax = plt.subplots(figsize=(PUB_DOUBLE_COL_WIDTH, 3.8))
        ax.text(0.5, 0.5, "No normalized evaluation metrics available.", ha="center")
        ax.set_axis_off()
        fig.tight_layout()
        output_path = output_dir_path / "ev_metrics_by_usage_category_heatmap.pdf"
        fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
        plt.close(fig)
        return {
            "long_csv": output_dir_path / "ev_metrics_by_usage_category_long.csv",
            "heatmap_pdf": output_path,
        }

    top_metrics = (
        counts.groupby("metric", as_index=False)
        .agg(total_count=("count", "sum"))
        .sort_values(["total_count", "metric"], ascending=[False, True])
        .head(top_n_metrics)["metric"]
        .tolist()
    )
    usage_categories = sorted(counts["Usage category"].unique())
    pivot = (
        counts[counts["metric"].isin(top_metrics)]
        .pivot(index="Usage category", columns="metric", values="count")
        .reindex(index=usage_categories, columns=top_metrics)
        .fillna(0)
        .astype(int)
    )
    pivot.to_csv(output_dir_path / "ev_metrics_by_usage_category_top_counts.csv")

    fig_width = max(PUB_DOUBLE_COL_WIDTH * 1.35, 1.15 * len(top_metrics) + 3.4)
    fig_height = max(2.6, 0.72 * len(usage_categories) + 1.7)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = LinearSegmentedColormap.from_list(
        "metric_blue_tint",
        ["#f4f8fc", "#d8e7f2", "#b1cde2", "#7fa8ca", PLOT_DEV_COLOR],
    )
    vmax = max(int(pivot.to_numpy().max()), 1)
    norm = Normalize(vmin=0, vmax=vmax)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(top_metrics)))
    ax.set_xticklabels([textwrap.fill(str(metric), width=16) for metric in top_metrics], rotation=35, ha="right")
    ax.set_yticks(range(len(usage_categories)))
    ax.set_yticklabels(usage_categories)
    ax.set_title("Evaluation metrics used by usage category")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Usage\ncategory")

    for i, usage_category in enumerate(usage_categories):
        for j, metric in enumerate(top_metrics):
            value = int(pivot.loc[usage_category, metric])
            if value == 0:
                continue
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=9.0,
                color="white" if value >= vmax * 0.55 else PLOT_TEXT_DARK,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Unique studies")
    fig.tight_layout()
    heatmap_path = output_dir_path / "ev_metrics_by_usage_category_heatmap.pdf"
    fig.savefig(heatmap_path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

    return {
        "long_csv": output_dir_path / "ev_metrics_by_usage_category_long.csv",
        "top_csv": output_dir_path / "ev_metrics_by_usage_category_top_counts.csv",
        "heatmap_pdf": heatmap_path,
    }


def _split_contextual_qualifier_labels(value: object) -> list[str]:
    if pd.isna(value):
        return []

    labels = []
    for label in re.split(r"[,;]", str(value)):
        clean_label = label.strip()
        if clean_label:
            labels.append(clean_label)
    return labels


def _contextual_qualifier_order_from_mapping(mapping_path: Path) -> list[str]:
    if not mapping_path.exists():
        return []

    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    ordered_labels: list[str] = []
    for mapped_value in mapping.values():
        for label in _split_contextual_qualifier_labels(mapped_value):
            if label not in ordered_labels:
                ordered_labels.append(label)
    return ordered_labels


def _study_contextual_qualifier_counts_by_usage_category(
    df: pd.DataFrame,
    qualifier_column: str = "Contextual qualifier(s)",
) -> pd.DataFrame:
    required_columns = ["Title", "Usage category", qualifier_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    subset = df[required_columns].copy()
    subset = subset.dropna(subset=["Title", "Usage category", qualifier_column])
    subset["study_id"] = subset["Title"].astype(str).str.strip()
    subset["usage_category"] = subset["Usage category"].astype(str).str.strip()
    subset = subset[(subset["study_id"] != "") & (subset["usage_category"] != "")]
    if subset.empty:
        return pd.DataFrame(
            columns=["Usage category", "qualifier", "count", "total_studies", "percentage"]
        )

    exploded_rows = []
    for _, row in subset.iterrows():
        study_id = row["study_id"]
        usage_category = row["usage_category"]
        for qualifier in _split_contextual_qualifier_labels(row[qualifier_column]):
            exploded_rows.append(
                {
                    "study_id": study_id,
                    "Usage category": usage_category,
                    "qualifier": qualifier,
                }
            )

    if not exploded_rows:
        return pd.DataFrame(
            columns=["Usage category", "qualifier", "count", "total_studies", "percentage"]
        )

    exploded = pd.DataFrame(exploded_rows).drop_duplicates(
        ["study_id", "Usage category", "qualifier"]
    )
    usage_totals = (
        exploded.groupby("Usage category", as_index=False)
        .agg(total_studies=("study_id", "nunique"))
    )
    counts = (
        exploded.groupby(["Usage category", "qualifier"], as_index=False)
        .agg(count=("study_id", "nunique"))
        .merge(usage_totals, on="Usage category", how="left")
        .sort_values(["Usage category", "count", "qualifier"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    counts["percentage"] = counts["count"] / counts["total_studies"] * 100.0
    counts.attrs["total_studies"] = int(exploded["study_id"].nunique())
    return counts


def plot_contextual_qualifier_matrix_by_usage_category(
    df: pd.DataFrame,
    output_dir: str = "contextual_qualifier_overview",
    mapping_path: str | Path = "raw_data_mappings/Contextual qualifier(s).json",
) -> dict[str, Path]:
    """
    Plot a paper-ready matrix of contextual qualifiers by NLP usage category.

    The plot uses the normalized ``Contextual qualifier(s)`` JSON mapping,
    splits multi-qualifier cells, and counts each study once per
    qualifier/usage-category pair.
    """

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(".matplotlib_cache").resolve()),
    )
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    _setup_publication_style()

    counts = _study_contextual_qualifier_counts_by_usage_category(df)
    long_csv = output_dir_path / "contextual_qualifiers_by_usage_category_long.csv"
    matrix_csv = output_dir_path / "contextual_qualifiers_by_usage_category_matrix.csv"
    counts.to_csv(long_csv, index=False)

    mapping_order = _contextual_qualifier_order_from_mapping(Path(mapping_path))
    observed_order = []
    if not counts.empty:
        observed_order = (
            counts.groupby("qualifier", as_index=False)
            .agg(total_count=("count", "sum"))
            .sort_values(["total_count", "qualifier"], ascending=[False, True])["qualifier"]
            .tolist()
        )
    observed_labels = set(observed_order)
    qualifier_order = [
        label for label in CONTEXTUAL_QUALIFIER_SENTIMENT_ORDER if label in observed_labels
    ]
    qualifier_order.extend(
        label for label in mapping_order if label in observed_labels and label not in qualifier_order
    )
    qualifier_order.extend(label for label in observed_order if label not in qualifier_order)

    if counts.empty or not qualifier_order:
        pd.DataFrame().to_csv(matrix_csv)
        fig, ax = plt.subplots(figsize=(PUB_DOUBLE_COL_WIDTH, 3.2))
        ax.text(0.5, 0.5, "No contextual qualifiers available.", ha="center")
        ax.set_axis_off()
        fig.tight_layout()
        output_path = output_dir_path / "contextual_qualifiers_by_usage_category_matrix.pdf"
        fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
        plt.close(fig)
        return {
            "long_csv": long_csv,
            "matrix_csv": matrix_csv,
            "matrix_pdf": output_path,
        }

    usage_categories = sorted(counts["Usage category"].unique())
    pivot = (
        counts.pivot(index="Usage category", columns="qualifier", values="count")
        .reindex(index=usage_categories, columns=qualifier_order)
        .fillna(0)
        .astype(int)
    )
    pivot.to_csv(matrix_csv)

    fig_width = max(PUB_DOUBLE_COL_WIDTH, 1.0 * len(qualifier_order) + 3.2)
    fig_height = max(2.6, 0.64 * len(usage_categories) + 1.7)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = LinearSegmentedColormap.from_list(
        "qualifier_green_tint",
        ["#f7fbf7", "#dcefdc", "#b7dcb8", "#79bd82", "#3f8f56"],
    )
    vmax = max(int(pivot.to_numpy().max()), 1)
    norm = Normalize(vmin=0, vmax=vmax)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(qualifier_order)))
    ax.set_xticklabels(
        [textwrap.fill(str(qualifier), width=15) for qualifier in qualifier_order],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(range(len(usage_categories)))
    ax.set_yticklabels(usage_categories)
    ax.set_title("Contextual qualifiers by NLP usage category")
    ax.set_xlabel("Contextual qualifier")
    ax.set_ylabel("NLP usage\ncategory")

    for i, usage_category in enumerate(usage_categories):
        for j, qualifier in enumerate(qualifier_order):
            value = int(pivot.loc[usage_category, qualifier])
            if value == 0:
                continue
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=9.0,
                color="white" if value >= vmax * 0.55 else PLOT_TEXT_DARK,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Unique studies")
    fig.tight_layout()
    matrix_pdf = output_dir_path / "contextual_qualifiers_by_usage_category_matrix.pdf"
    fig.savefig(matrix_pdf, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

    return {
        "long_csv": long_csv,
        "matrix_csv": matrix_csv,
        "matrix_pdf": matrix_pdf,
    }


def _draw_combined_eval_size_histogram_ax(ax, parsed: pd.DataFrame) -> None:
    from matplotlib.ticker import MaxNLocator

    unique_parsed = _unique_eval_sample_sizes_by_study(parsed)
    text_values = unique_parsed.loc[unique_parsed["unit"] == "texts", "sample size"].dropna()
    patient_values = unique_parsed.loc[unique_parsed["unit"] == "patients", "sample size"].dropna()
    combined_values = unique_parsed["sample size"].dropna()
    text_studies = (
        unique_parsed.loc[unique_parsed["unit"] == "texts", "Title"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .nunique()
    )
    patient_studies = (
        unique_parsed.loc[unique_parsed["unit"] == "patients", "Title"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .nunique()
    )

    if combined_values.empty:
        ax.text(0.5, 0.5, "No parsed evaluation sample sizes.", ha="center")
        ax.set_axis_off()
        return

    bins = _log_hist_bins(combined_values)
    if not text_values.empty:
        ax.hist(
            text_values,
            bins=bins,
            color=PLOT_DEV_COLOR,
            alpha=0.55,
            edgecolor="white",
            label=(
                f"Texts (studies={text_studies}, "
                f"median={text_values.median():.0f})"
            ),
        )
    if not patient_values.empty:
        ax.hist(
            patient_values,
            bins=bins,
            color=PLOT_EVAL_COLOR,
            alpha=0.50,
            edgecolor="white",
            label=(
                f"Patients (studies={patient_studies}, "
                f"median={patient_values.median():.0f})"
            ),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Evaluation sample size (log scale)")
    ax.set_ylabel("Number of studies")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8.8)


def _draw_ev_metric_heatmap_ax(
    ax,
    df: pd.DataFrame,
    top_n_metrics: int = 16,
    xtick_wrap_width: int = 16,
    ytick_wrap_width: int | None = None,
):
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    counts = _study_metric_counts_by_usage_category(df, "Ev metrics")
    if counts.empty:
        ax.text(0.5, 0.5, "No normalized evaluation metrics available.", ha="center")
        ax.set_axis_off()
        return None

    top_metrics = (
        counts.groupby("metric", as_index=False)
        .agg(total_count=("count", "sum"))
        .sort_values(["total_count", "metric"], ascending=[False, True])
        .head(top_n_metrics)["metric"]
        .tolist()
    )
    usage_categories = sorted(counts["Usage category"].unique())
    pivot = (
        counts[counts["metric"].isin(top_metrics)]
        .pivot(index="Usage category", columns="metric", values="count")
        .reindex(index=usage_categories, columns=top_metrics)
        .fillna(0)
        .astype(int)
    )

    cmap = LinearSegmentedColormap.from_list(
        "metric_blue_tint",
        ["#f4f8fc", "#d8e7f2", "#b1cde2", "#7fa8ca", PLOT_DEV_COLOR],
    )
    vmax = max(int(pivot.to_numpy().max()), 1)
    norm = Normalize(vmin=0, vmax=vmax)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(top_metrics)))
    ax.set_xticklabels(
        [textwrap.fill(str(metric), width=xtick_wrap_width) for metric in top_metrics],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(range(len(usage_categories)))
    ytick_labels = (
        [textwrap.fill(str(category), width=ytick_wrap_width) for category in usage_categories]
        if ytick_wrap_width is not None
        else usage_categories
    )
    ax.set_yticklabels(ytick_labels)
    ax.set_title("Evaluation metrics by usage category")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Usage category")

    for i, usage_category in enumerate(usage_categories):
        for j, metric in enumerate(top_metrics):
            value = int(pivot.loc[usage_category, metric])
            if value == 0:
                continue
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if value >= vmax * 0.55 else PLOT_TEXT_DARK,
                fontweight="bold",
            )
    return im


def plot_evaluation_metrics_sample_size_panel(
    df: pd.DataFrame,
    output_dir: str = "paper_panels",
    top_n_metrics: int = 10,
) -> dict[str, Path]:
    """
    Create a compact paper-ready panel combining evaluation sample sizes and
    model evaluation metrics by NLP usage category.
    """

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(".matplotlib_cache").resolve()),
    )
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    _setup_publication_style()

    parsed = _unique_eval_sample_sizes_by_study(parse_eval_sample_sizes(df))

    fig = plt.figure(figsize=(PUB_DOUBLE_COL_WIDTH, 6.25), constrained_layout=False)
    gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[0.86, 1.35],
        hspace=0.54,
    )

    ax_hist = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])

    _draw_combined_eval_size_histogram_ax(ax_hist, parsed)
    ax_hist.set_title("A. Evaluation sample sizes", loc="left", fontweight="bold")

    im = _draw_ev_metric_heatmap_ax(
        ax_heat,
        df,
        top_n_metrics=top_n_metrics,
        xtick_wrap_width=12,
        ytick_wrap_width=24,
    )
    ax_heat.set_title("", loc="center")
    ax_heat.set_title("B. Evaluation metrics by NLP usage category", loc="left", fontweight="bold")
    ax_heat.set_ylabel("NLP usage\ncategory")
    if im is not None:
        fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.02, label="Unique studies")

    fig.subplots_adjust(
        left=0.22,
        right=0.91,
        bottom=0.20,
        top=0.94,
        hspace=0.72,
    )
    pdf_path = output_dir_path / "evaluation_metrics_sample_size_panel.pdf"
    png_path = output_dir_path / "evaluation_metrics_sample_size_panel.png"
    fig.savefig(pdf_path, dpi=PUB_DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)
    return {
        "pdf": pdf_path,
        "png": png_path,
    }


def _build_comparison_graph_for_usage_category(
    pairwise_df: pd.DataFrame,
    usage_category: str,
    weight_mode: str = "study",
    include_self_edges: bool = False,
):
    nodes, comparisons = _prepare_comparison_graph_rows(
        pairwise_df,
        "architecture",
        include_self_edges=True,
    )
    comparisons = comparisons[comparisons["usage category 1"] == usage_category].copy()
    if comparisons.empty:
        return None, []

    comparisons = comparisons[comparisons["metric value 1"] != comparisons["metric value 2"]].copy()
    comparisons["winner"] = comparisons.apply(
        lambda row: row["node 1"]
        if row["metric value 1"] > row["metric value 2"]
        else row["node 2"],
        axis=1,
    )
    comparisons["loser"] = comparisons.apply(
        lambda row: row["node 2"]
        if row["metric value 1"] > row["metric value 2"]
        else row["node 1"],
        axis=1,
    )
    if not include_self_edges:
        comparisons = comparisons[comparisons["winner"] != comparisons["loser"]].copy()

    edge_records = []
    for (winner, loser), edge_df in comparisons.groupby(["winner", "loser"], sort=True):
        study_titles = set()
        for row_dict in edge_df.to_dict(orient="records"):
            for title_key in ("Title 1", "Title 2", "Title"):
                study_title = _normalize_docx_text(row_dict.get(title_key, ""))
                if study_title:
                    study_titles.add(study_title)
        comparison_weight = int(len(edge_df))
        study_weight = int(len(study_titles)) if study_titles else comparison_weight
        edge_records.append({
            "source": winner,
            "target": loser,
            "comparison_weight": comparison_weight,
            "study_weight": study_weight,
            "weight": study_weight if weight_mode == "study" else comparison_weight,
        })

    ordered_nodes = _weighted_circular_node_order(
        sorted(set(comparisons["node 1"].dropna()) | set(comparisons["node 2"].dropna())),
        edge_records,
    )
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "The comparison graph panel requires networkx. Install it with `pip install networkx`."
        ) from exc

    graph = nx.DiGraph()
    graph.add_nodes_from(ordered_nodes)
    for edge in edge_records:
        graph.add_edge(edge["source"], edge["target"], **edge)
    return graph, edge_records


def _draw_comparison_graph_ax(
    ax,
    graph,
    edge_weights: list[dict[str, object]],
    title: str,
    subtitle: str,
    graph_scale: float = 1.65,
    view_limit: float = 2.55,
    node_size_scale: float = 1.0,
    arrowhead_scale: float = 1.0,
    labels_outside: bool = False,
    label_offset: float = 0.22,
) -> None:
    import networkx as nx
    from matplotlib.patches import FancyArrowPatch

    if graph is None or graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No comparison graph available.", ha="center", va="center")
        ax.set_axis_off()
        return

    ordered_nodes = list(graph.nodes)
    pos = {}
    for idx, node in enumerate(ordered_nodes):
        angle = -math.pi / 2 + (2 * math.pi * idx / max(1, len(ordered_nodes)))
        pos[node] = (
            math.cos(angle) * graph_scale,
            math.sin(angle) * graph_scale,
        )
    node_sizes = [
        (1940
        + 240
        * (
            graph.in_degree(node, weight="weight")
            + graph.out_degree(node, weight="weight")
        ) ** 0.5
        + max(0, len(str(node)) - 14) * 38
        ) * node_size_scale
        for node in graph.nodes
    ]
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

    def arc_label_geometry(start, end, rad: float, t: float) -> tuple[float, float, float]:
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

    weights = [int(data.get("weight", 0)) for _, _, data in graph.edges(data=True)]
    min_log_weight = min((math.log1p(weight) for weight in weights), default=0)
    max_log_weight = max((math.log1p(weight) for weight in weights), default=0)

    def edge_strength(weight: int) -> float:
        if max_log_weight == min_log_weight:
            return 1.0
        return (math.log1p(weight) - min_log_weight) / (max_log_weight - min_log_weight)

    def edge_width(weight: int) -> float:
        return 0.70 + 2.85 * edge_strength(weight)

    def edge_alpha(weight: int) -> float:
        return 0.18 + 0.72 * edge_strength(weight)

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
                color=PLOT_INK_COLOR,
                alpha=alpha,
                clip_on=False,
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
            color=PLOT_INK_COLOR,
            alpha=alpha,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
            zorder=1,
        )
        ax.add_patch(patch)

        label_x, label_y, label_angle = arc_label_geometry(start, end, rad, t=0.78)
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

    for source, target, data in sorted(
        graph.edges(data=True),
        key=lambda item: (
            int(item[2].get("weight", 0)),
            str(item[0]),
            str(item[1]),
        ),
    ):
        draw_weighted_edge(source, target, int(data.get("weight", 0)))

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color="#d9e9f5",
        edgecolors="#234",
        ax=ax,
    )
    for node in graph.nodes:
        x, y = pos[node]
        node_lines, font_size = _node_label_style(node, width=18)
        if labels_outside:
            norm = math.hypot(x, y) or 1.0
            unit_x, unit_y = x / norm, y / norm
            text_x = x + unit_x * label_offset
            text_y = y + unit_y * label_offset
            ha = "left" if unit_x >= 0 else "right"
            va = "bottom" if unit_y >= 0 else "top"
            ax.text(
                text_x,
                text_y,
                "\n".join(node_lines),
                fontsize=font_size,
                fontweight="bold",
                ha=ha,
                va=va,
                linespacing=0.90,
                bbox=label_bbox,
                zorder=5,
            )
        else:
            ax.text(
                x,
                y,
                "\n".join(node_lines),
                fontsize=font_size,
                fontweight="bold",
                ha="center",
                va="center",
                linespacing=0.90,
                zorder=5,
            )
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(-view_limit, view_limit)
    ax.set_ylim(-(view_limit - 0.10), (view_limit - 0.10))


def plot_metrics_eval_sizes_ie_panel(
    df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    output_dir: str = "paper_panels",
) -> dict[str, Path]:
    """
    Create a single publication-ready panel combining:
      - combined evaluation sample sizes
      - evaluation metric heatmap by usage category
      - study-weighted information-extraction comparison graph
    """

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(".matplotlib_cache").resolve()),
    )
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    _setup_publication_style()

    parsed = _unique_eval_sample_sizes_by_study(parse_eval_sample_sizes(df))
    graph, edge_records = _build_comparison_graph_for_usage_category(
        pairwise_df,
        "Information extraction",
        weight_mode="study",
        include_self_edges=False,
    )

    fig = plt.figure(figsize=(PUB_DOUBLE_COL_WIDTH * 2.60, 9.8))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.22, 0.78],
        width_ratios=[1.12, 0.88],
        hspace=0.26,
        wspace=0.18,
    )

    ax_graph = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, :])

    _draw_comparison_graph_ax(
        ax_graph,
        graph,
        edge_records,
        "",
        "",
        graph_scale=1.70,
        view_limit=2.68,
        node_size_scale=0.42,
        arrowhead_scale=1.26,
        labels_outside=True,
        label_offset=0.21,
    )

    _draw_combined_eval_size_histogram_ax(ax_hist, parsed)
    ax_hist.set_title("Evaluation sample sizes")

    im = _draw_ev_metric_heatmap_ax(ax_heat, df)
    if im is not None:
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.03, label="Unique studies")

    fig.suptitle(
        "Evaluation sizes, metrics, and study-weighted comparison graph",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_path = output_dir_path / "evaluation_metrics_sizes_ie_panel.pdf"
    fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)
    return {
        "pdf": output_path,
    }


def _study_province_counts(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    required_columns = ["Title", column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    subset = df[required_columns].copy()
    subset = subset.dropna(subset=["Title", column])
    subset["study_id"] = subset["Title"].astype(str).str.strip()
    subset["region"] = subset[column].astype(str).str.strip()
    subset = subset[(subset["study_id"] != "") & (subset["region"] != "")]
    if subset.empty:
        return pd.DataFrame(columns=["province", "count", "percentage"]), 0

    exploded_rows = []
    national_like = {"netherlands", "not reported", "nr", "unassigned", "other"}
    for _, row in subset.iterrows():
        study_id = row["study_id"]
        for token in str(row["region"]).split(","):
            cleaned = token.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            province = REGION_TO_PROVINCE_ALIASES.get(lowered, cleaned)
            if province.lower() in national_like:
                continue
            exploded_rows.append({"study_id": study_id, "province": province})

    if not exploded_rows:
        return pd.DataFrame(columns=["province", "count", "percentage"]), 0

    exploded = pd.DataFrame(exploded_rows).drop_duplicates(["study_id", "province"])
    total_mapped_studies = exploded["study_id"].nunique()
    counts = (
        exploded.groupby("province", as_index=False)
        .agg(count=("study_id", "nunique"))
        .sort_values(["count", "province"], ascending=[False, True])
        .reset_index(drop=True)
    )
    counts["percentage"] = (
        counts["count"] / total_mapped_studies * 100.0 if total_mapped_studies else 0.0
    )
    counts.attrs["total_mapped_studies"] = int(total_mapped_studies)
    return counts, int(total_mapped_studies)


def _study_province_counts_any(
    df: pd.DataFrame,
    columns: tuple[str, ...] = ("Dev region", "Ev region"),
) -> pd.DataFrame:
    required_columns = ["Title", "Dev region", "Ev region"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    subset = df[required_columns].copy()
    subset = subset.dropna(subset=["Title"])
    subset["study_id"] = subset["Title"].astype(str).str.strip()
    subset = subset[subset["study_id"] != ""]
    if subset.empty:
        return pd.DataFrame(columns=["country", "province", "count", "percentage"])

    def classify_token(token: str, context: str) -> tuple[str, str] | None:
        token = str(token).strip()
        if not token:
            return None
        lowered = token.lower()
        if lowered in {"netherlands", "not reported", "nr"}:
            return None
        if token in {"North Holland", "South Holland", "Utrecht", "Groningen", "Gelderland", "North Brabant", "Overijssel", "Friesland", "Drenthe", "Flevoland", "Zeeland"}:
            return ("NLD", token)
        if token in {"Antwerp", "East Flanders", "West Flanders", "Flemish Brabant", "Walloon Brabant", "Hainaut", "Namur", "Liège", "Liege", "Luxembourg"}:
            province = "Liège" if token == "Liege" else token
            return ("BEL", province)
        if token == "Limburg":
            belgian_context = any(
                marker in context.lower()
                for marker in [
                    "antwerp",
                    "east flanders",
                    "west flanders",
                    "flemish brabant",
                    "walloon brabant",
                    "hainaut",
                    "namur",
                    "liège",
                    "liege",
                    "luxembourg",
                    "belgium",
                ]
            )
            return ("BEL", "Limburg") if belgian_context else ("NLD", "Limburg")
        return None

    exploded_rows = []
    for _, row in subset.iterrows():
        study_id = row["study_id"]
        context = " ".join(
            str(row.get(col, ""))
            for col in columns
            if col in row.index and pd.notna(row.get(col, ""))
        )
        seen_pairs = set()
        for column in columns:
            value = row.get(column, "")
            if pd.isna(value):
                continue
            for token in str(value).split(","):
                classified = classify_token(token.strip(), context)
                if classified is None or classified in seen_pairs:
                    continue
                seen_pairs.add(classified)
                country, province = classified
                exploded_rows.append(
                    {"study_id": study_id, "country": country, "province": province}
                )

    if not exploded_rows:
        return pd.DataFrame(columns=["country", "province", "count", "percentage"])

    exploded = pd.DataFrame(exploded_rows).drop_duplicates(["study_id", "country", "province"])
    total_studies = exploded["study_id"].nunique()
    counts = (
        exploded.groupby(["country", "province"], as_index=False)
        .agg(count=("study_id", "nunique"))
        .sort_values(["country", "count", "province"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    counts["percentage"] = counts["count"] / total_studies * 100.0 if total_studies else 0.0
    counts.attrs["total_studies"] = int(total_studies)
    return counts


def _province_layout(country: str) -> dict[str, tuple[float, float, float, float]]:
    if country == "Netherlands":
        return {
            "Groningen": (2.75, 0.10, 0.95, 0.65),
            "Friesland": (1.70, 0.10, 1.05, 0.75),
            "Drenthe": (2.00, 0.78, 0.95, 0.65),
            "North Holland": (0.65, 1.10, 0.95, 1.05),
            "Flevoland": (1.65, 1.15, 0.82, 0.58),
            "Overijssel": (2.55, 1.38, 0.98, 0.90),
            "Utrecht": (1.60, 1.95, 0.75, 0.55),
            "South Holland": (0.55, 2.20, 1.25, 0.90),
            "Gelderland": (2.05, 2.25, 1.20, 1.00),
            "Zeeland": (0.10, 3.30, 0.88, 0.78),
            "North Brabant": (1.15, 3.35, 1.45, 0.95),
            "Limburg": (2.72, 3.45, 0.78, 0.88),
        }
    if country == "Belgium":
        return {
            "West Flanders": (0.45, 0.95, 0.95, 0.75),
            "East Flanders": (1.30, 0.80, 0.95, 0.80),
            "Antwerp": (2.20, 0.72, 0.88, 0.78),
            "Limburg": (3.08, 0.86, 0.82, 0.80),
            "Flemish Brabant": (1.95, 1.65, 0.98, 0.72),
            "Walloon Brabant": (1.95, 2.55, 0.92, 0.62),
            "Hainaut": (0.95, 2.45, 1.00, 0.78),
            "Namur": (2.75, 2.95, 0.90, 0.72),
            "Liège": (3.25, 2.55, 0.90, 0.72),
            "Luxembourg": (2.95, 3.78, 1.05, 0.72),
        }
    raise ValueError(f"Unsupported country: {country}")


def _country_outline(country: str) -> list[tuple[float, float]]:
    if country == "Netherlands":
        return [
            (0.20, 0.05),
            (3.82, 0.05),
            (3.92, 1.00),
            (3.55, 1.95),
            (3.65, 3.10),
            (3.05, 4.55),
            (1.05, 4.75),
            (0.15, 3.90),
            (0.05, 1.85),
        ]
    if country == "Belgium":
        return [
            (0.20, 0.40),
            (3.95, 0.35),
            (4.08, 1.55),
            (3.88, 2.65),
            (3.50, 3.85),
            (2.30, 4.60),
            (0.95, 4.25),
            (0.20, 3.15),
            (0.08, 1.50),
        ]
    raise ValueError(f"Unsupported country: {country}")


def _province_polygon(x: float, y: float, w: float, h: float) -> list[tuple[float, float]]:
    return [
        (x + 0.05 * w, y + 0.02 * h),
        (x + 0.86 * w, y + 0.00 * h),
        (x + 0.98 * w, y + 0.22 * h),
        (x + 1.00 * w, y + 0.82 * h),
        (x + 0.75 * w, y + 1.00 * h),
        (x + 0.14 * w, y + 0.96 * h),
        (x + 0.00 * w, y + 0.72 * h),
        (x + 0.00 * w, y + 0.18 * h),
    ]


def _plot_province_map_panel(
    ax,
    counts: pd.DataFrame,
    country: str,
    title: str,
    cmap,
    vmax: int,
) -> None:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    cartopy.config["data_dir"] = str(Path(".cartopy_cache").resolve())
    Path(cartopy.config["data_dir"]).mkdir(parents=True, exist_ok=True)

    if not isinstance(ax.projection, ccrs.CRS):
        raise TypeError("Province map panel requires a cartopy GeoAxes.")

    reader = shpreader.Reader(
        shpreader.natural_earth(
            resolution="10m",
            category="cultural",
            name="admin_1_states_provinces",
        )
    )
    records = []
    for rec in reader.records():
        attrs = rec.attributes
        if attrs.get("adm0_a3") != ("NLD" if country == "Netherlands" else "BEL"):
            continue
        if attrs.get("type_en") not in {"Province", "Capital Region"}:
            continue
        records.append(rec)

    if not records:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No province geometries available.", ha="center", va="center")
        ax.set_axis_off()
        return

    count_lookup = {str(row["province"]): int(row["count"]) for _, row in counts.iterrows()}
    norm = Normalize(vmin=0, vmax=max(vmax, 1))
    count_values = []

    for rec in records:
        province = rec.attributes.get("name_en") or rec.attributes.get("name")
        if province == "Brussels Capital":
            continue
        if country == "Belgium" and province == "Brussels":
            continue
        if country == "Netherlands" and province in {"Bonaire", "St. Eustatius", "Saba"}:
            continue
        count = int(count_lookup.get(str(province), 0))
        count_values.append(count)
        ax.add_geometries(
            [rec.geometry],
            ccrs.PlateCarree(),
            facecolor=cmap(norm(count)),
            edgecolor="#ffffff",
            linewidth=0.9,
            zorder=2,
        )

        centroid = rec.geometry.representative_point()
        ax.text(
            centroid.x,
            centroid.y,
            f"{province}\n{count}",
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold" if count else "normal",
            color=PLOT_TEXT_DARK,
            zorder=3,
        )

    all_geoms = [rec.geometry for rec in records]
    minx = min(geom.bounds[0] for geom in all_geoms)
    miny = min(geom.bounds[1] for geom in all_geoms)
    maxx = max(geom.bounds[2] for geom in all_geoms)
    maxy = max(geom.bounds[3] for geom in all_geoms)
    xpad = max((maxx - minx) * 0.07, 0.15)
    ypad = max((maxy - miny) * 0.07, 0.15)
    ax.set_extent(
        [minx - xpad, maxx + xpad, miny - ypad, maxy + ypad],
        crs=ccrs.PlateCarree(),
    )
    ax.set_facecolor(PLOT_BG_COLOR)
    ax.set_title(title)


def _plot_combined_province_map(
    ax,
    counts: pd.DataFrame,
    title: str,
    cmap,
    vmax: int,
) -> None:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    cartopy.config["data_dir"] = str(Path(".cartopy_cache").resolve())
    Path(cartopy.config["data_dir"]).mkdir(parents=True, exist_ok=True)

    if not isinstance(ax.projection, ccrs.CRS):
        raise TypeError("Province map panel requires a cartopy GeoAxes.")

    reader = shpreader.Reader(
        shpreader.natural_earth(
            resolution="10m",
            category="cultural",
            name="admin_1_states_provinces",
        )
    )
    records = []
    for rec in reader.records():
        attrs = rec.attributes
        if attrs.get("adm0_a3") not in {"NLD", "BEL"}:
            continue
        if attrs.get("type_en") not in {"Province", "Capital Region"}:
            continue
        records.append(rec)

    if not records:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No province geometries available.", ha="center", va="center")
        ax.set_axis_off()
        return

    count_lookup = {
        (str(row["country"]), str(row["province"])): int(row["count"])
        for _, row in counts.iterrows()
    }
    norm = Normalize(vmin=0, vmax=max(vmax, 1))

    all_geoms = []
    country_centroids: dict[str, list[tuple[float, float]]] = {"NLD": [], "BEL": []}

    for rec in records:
        country = rec.attributes.get("adm0_a3")
        province = rec.attributes.get("name_en") or rec.attributes.get("name")
        if country == "BEL" and province == "Brussels Capital":
            continue
        if country == "NLD" and province in {"Bonaire", "St. Eustatius", "Saba"}:
            continue
        count = int(count_lookup.get((str(country), str(province)), 0))
        all_geoms.append(rec.geometry)
        centroid = rec.geometry.representative_point()
        country_centroids.setdefault(country, []).append((centroid.x, centroid.y))
        ax.add_geometries(
            [rec.geometry],
            ccrs.PlateCarree(),
            facecolor=cmap(norm(count)),
            edgecolor="#ffffff",
            linewidth=0.9,
            zorder=2,
        )
        ax.text(
            centroid.x,
            centroid.y,
            f"{province}\n{count}",
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="bold" if count else "normal",
            color=PLOT_TEXT_DARK,
            zorder=3,
        )

    if all_geoms:
        minx = min(geom.bounds[0] for geom in all_geoms)
        miny = min(geom.bounds[1] for geom in all_geoms)
        maxx = max(geom.bounds[2] for geom in all_geoms)
        maxy = max(geom.bounds[3] for geom in all_geoms)
        xpad = max((maxx - minx) * 0.06, 0.15)
        ypad = max((maxy - miny) * 0.06, 0.15)
        ax.set_extent(
            [minx - xpad, maxx + xpad, miny - ypad, maxy + ypad],
            crs=ccrs.PlateCarree(),
        )

    if country_centroids["NLD"]:
        xs, ys = zip(*country_centroids["NLD"], strict=False)
        ax.text(
            sum(xs) / len(xs),
            sum(ys) / len(ys) + 0.35,
            "Netherlands",
            transform=ccrs.PlateCarree(),
            ha="center",
            va="bottom",
            fontsize=11.0,
            fontweight="bold",
            color=PLOT_DEV_COLOR,
            zorder=4,
        )
    if country_centroids["BEL"]:
        xs, ys = zip(*country_centroids["BEL"], strict=False)
        ax.text(
            sum(xs) / len(xs),
            sum(ys) / len(ys) + 0.35,
            "Belgium",
            transform=ccrs.PlateCarree(),
            ha="center",
            va="bottom",
            fontsize=11.0,
            fontweight="bold",
            color=PLOT_EVAL_COLOR,
            zorder=4,
        )

    ax.set_facecolor(PLOT_BG_COLOR)
    ax.set_title(title)


def plot_region_study_province_maps(
    df: pd.DataFrame,
    output_dir: str = "region_study_province_maps",
) -> dict[str, Path]:
    """
    Plot study counts by province on a single combined Netherlands/Belgium map.

    Studies can contribute to multiple provinces when the region field contains
    multiple comma-separated locations.
    """

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(".matplotlib_cache").resolve()),
    )
    import cartopy
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cartopy.config["data_dir"] = str(Path(".cartopy_cache").resolve())
    Path(cartopy.config["data_dir"]).mkdir(parents=True, exist_ok=True)

    _setup_publication_style()

    province_cmap = LinearSegmentedColormap.from_list(
        "province_blue_tint",
        ["#f4f8fc", "#d8e7f2", "#b1cde2", "#7fa8ca", PLOT_DEV_COLOR],
    )

    dev_counts, dev_mapped_total = _study_province_counts(df, "Dev region")
    ev_counts, ev_mapped_total = _study_province_counts(df, "Ev region")
    dev_map_counts = _study_province_counts_any(df, ("Dev region",))
    ev_map_counts = _study_province_counts_any(df, ("Ev region",))
    combined_counts = _study_province_counts_any(df)

    dev_counts.to_csv(output_dir_path / "dev_region_province_counts.csv", index=False)
    ev_counts.to_csv(output_dir_path / "ev_region_province_counts.csv", index=False)
    combined_counts.to_csv(output_dir_path / "province_counts_combined_nl_be.csv", index=False)

    vmax = int(combined_counts["count"].max() if not combined_counts.empty else 0)
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(PUB_DOUBLE_COL_WIDTH * 1.55, 6.0),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_facecolor(PLOT_PANEL_BG_COLOR)
    _plot_combined_province_map(
        ax,
        combined_counts,
        f"Study counts by province [n={combined_counts.attrs.get('total_studies', 0)}]",
        province_cmap,
        vmax,
    )

    sm = plt.cm.ScalarMappable(cmap=province_cmap, norm=Normalize(vmin=0, vmax=max(vmax, 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Number of unique studies")
    fig.suptitle(
        "Study counts by province (Netherlands + Belgium)\n"
        "Comma-separated region labels are split before counting; nationwide/unassigned labels are excluded from province allocation.",
        y=0.995,
    )
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.03, right=0.92)
    output_path = output_dir_path / "study_counts_by_province_combined.pdf"
    fig.savefig(output_path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)
    panel_fig, panel_axes = plt.subplots(
        1,
        3,
        figsize=(PUB_DOUBLE_COL_WIDTH * 2.1, 6.1),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    panel_specs = [
        ("Development region", dev_map_counts),
        ("Evaluation region", ev_map_counts),
        ("Development + evaluation", combined_counts),
    ]
    panel_vmax = max(
        [int(dev_map_counts["count"].max() if not dev_map_counts.empty else 0)]
        + [int(ev_map_counts["count"].max() if not ev_map_counts.empty else 0)]
        + [int(combined_counts["count"].max() if not combined_counts.empty else 0)]
    )
    for ax, (label, panel_counts) in zip(panel_axes.flat, panel_specs, strict=False):
        ax.set_facecolor(PLOT_PANEL_BG_COLOR)
        _plot_combined_province_map(
            ax,
            panel_counts,
            f"{label} [n={panel_counts.attrs.get('total_studies', 0)}]",
            province_cmap,
            panel_vmax,
        )
    panel_sm = plt.cm.ScalarMappable(
        cmap=province_cmap,
        norm=Normalize(vmin=0, vmax=max(panel_vmax, 1)),
    )
    panel_sm.set_array([])
    panel_cbar = panel_fig.colorbar(panel_sm, ax=panel_axes.ravel().tolist(), fraction=0.03, pad=0.02)
    panel_cbar.set_label("Number of unique studies")
    panel_fig.suptitle(
        "Study counts by province (Netherlands + Belgium)\n"
        "Comma-separated region labels are split before counting; nationwide/unassigned labels are excluded from province allocation.",
        y=0.995,
    )
    panel_fig.subplots_adjust(top=0.90, bottom=0.06, left=0.02, right=0.93, wspace=0.02)
    panel_output_path = output_dir_path / "study_counts_by_province_panel.pdf"
    panel_fig.savefig(panel_output_path, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(panel_fig)
    return {
        "dev_csv": output_dir_path / "dev_region_province_counts.csv",
        "ev_csv": output_dir_path / "ev_region_province_counts.csv",
        "combined_csv": output_dir_path / "province_counts_combined_nl_be.csv",
        "pdf": output_path,
        "panel_pdf": panel_output_path,
    }


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
    colors = _architecture_color_map(categories)

    bottom = pd.Series(0.0, index=percentage_df.index)
    for category in categories:
        values = percentage_df[category]
        ax.bar(
            years,
            values,
            bottom=bottom,
            label=category,
            color=colors.get(category, "#8ba6c6"),
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
    colors = _architecture_color_map(categories)

    bottom = pd.Series(0, index=count_df.index, dtype=float)
    for category in categories:
        values = count_df[category].fillna(0)
        ax.bar(
            years,
            values,
            bottom=bottom,
            label=category,
            color=colors.get(category, "#8ba6c6"),
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
    output_name: str = "architecture_study_counts_by_year_panels.pdf",
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

    panels: list[tuple[str, pd.DataFrame, int]] = []
    for usage_category in sorted(df["Usage category"].dropna().unique()):
        counts = _architecture_study_counts_by_year(df, usage_category=usage_category)
        study_count = int(
            df.loc[
                df["Usage category"] == usage_category,
                "Title",
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )
        panels.append((usage_category, counts, study_count))

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
            for _, panel_df, _ in panels
            for architecture in panel_df.columns
        }
    )
    colors = _architecture_color_map(categories)

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
    for ax, (usage_category, count_df, study_count) in zip(axes_list, panels):
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
                color=colors.get(category, "#8ba6c6"),
                width=0.82,
                label=category,
            )
            bottom = bottom + values
            if category not in legend_handles:
                legend_handles[category] = bar[0]

        ax.set_title(f"{usage_category} [n={study_count}]")
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
        output_dir_path / "architecture_study_percentages_by_year_overall.pdf",
        "Model architecture usage by year (studies)",
        ylabel="Percentage of studies",
    )
    overall_counts = _architecture_study_counts_by_year(df)
    overall_counts.to_csv(output_dir_path / "architecture_study_counts_by_year_overall.csv")
    _plot_stacked_count_bars(
        overall_counts,
        output_dir_path / "architecture_study_counts_by_year_overall.pdf",
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
                output_dir_path / f"architecture_study_percentages_by_year_{safe_category}.pdf",
                f"Model architecture usage by year (studies): {usage_category}",
                ylabel="Percentage of studies",
            )
            counts.to_csv(
                output_dir_path / f"architecture_study_counts_by_year_{safe_category}.csv"
            )
            _plot_stacked_count_bars(
                counts,
                output_dir_path / f"architecture_study_counts_by_year_{safe_category}.pdf",
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
        col
        for col in [
            "Main metric",
            "Ev metrics",
            "Title",
            "First author",
            "Year",
            BASE_MODEL_COLUMN,
            "Base model",
            "Category",
            "Type of model",
        ]
        if col in df.columns
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
                if col in {"Title", "First author", "Year"}:
                    edge[f"{col} 1"] = _json_safe_scalar(row_1[col])
                    edge[f"{col} 2"] = _json_safe_scalar(row_2[col])
                    continue
                if col in {BASE_MODEL_COLUMN, "Base model", "Category", "Type of model"}:
                    if col in {BASE_MODEL_COLUMN, "Base model"}:
                        edge["base model 1"] = _json_safe_scalar(row_1[col])
                        edge["base model 2"] = _json_safe_scalar(row_2[col])
                    else:
                        edge[f"{col} 1"] = _json_safe_scalar(row_1[col])
                        edge[f"{col} 2"] = _json_safe_scalar(row_2[col])
                    continue
                edge[col] = _json_safe_scalar(row_1[col])

            for field in ("Title", "First author", "Year"):
                if field in row_1.index and field in row_2.index:
                    edge[f"{field} 1"] = _json_safe_scalar(row_1[field])
                    edge[f"{field} 2"] = _json_safe_scalar(row_2[field])
            doi_1 = row_1.get("DOI", row_1.get("Study DOI / link", ""))
            doi_2 = row_2.get("DOI", row_2.get("Study DOI / link", ""))
            doi_display_1, doi_href_1, doi_title_1 = _normalize_doi(doi_1)
            doi_display_2, doi_href_2, doi_title_2 = _normalize_doi(doi_2)
            edge["doi_display 1"] = _json_safe_scalar(doi_display_1)
            edge["doi_href 1"] = _json_safe_scalar(doi_href_1)
            edge["doi_title 1"] = _json_safe_scalar(doi_title_1)
            edge["doi_display 2"] = _json_safe_scalar(doi_display_2)
            edge["doi_href 2"] = _json_safe_scalar(doi_href_2)
            edge["doi_title 2"] = _json_safe_scalar(doi_title_2)

            edges.append(edge)

    numbered_optional_columns = []
    for col in optional_columns:
        if col in {"Title", "First author", "Year"}:
            numbered_optional_columns.extend([f"{col} 1", f"{col} 2"])
            continue
        if col in {BASE_MODEL_COLUMN, "Base model", "Category", "Type of model"}:
            if col in {BASE_MODEL_COLUMN, "Base model"}:
                numbered_optional_columns.extend(["base model 1", "base model 2"])
                continue
            numbered_optional_columns.extend([f"{col} 1", f"{col} 2"])
        else:
            numbered_optional_columns.append(col)
    numbered_optional_columns.extend(
        [
            "doi_display 1",
            "doi_href 1",
            "doi_title 1",
            "doi_display 2",
            "doi_href 2",
            "doi_title 2",
        ]
    )
    columns = [
        "Comparator block",
        *numbered_optional_columns,
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


def _comparison_mode_config() -> dict[str, dict[str, object]]:
    return {
        "architecture": {
            "label": "Architecture categories",
            "node_field": "model architecture category",
            "filter_categories": None,
            "detail_field": "Model architecture category",
        },
        "pretrained_base": {
            "label": "Base models: pretrained transformers",
            "node_field": "base model",
            "filter_categories": {PRETRAINED_TRANSFORMER_CATEGORY},
            "detail_field": "Base model",
        },
        "prompted_base": {
            "label": "Base models: prompted LLMs",
            "node_field": "base model",
            "filter_categories": PROMPTED_LLM_CATEGORIES,
            "detail_field": "Base model",
        },
    }


def _prepare_comparison_graph_rows(
    pairwise_df: pd.DataFrame,
    mode: str,
    include_self_edges: bool = False,
) -> tuple[list[str], pd.DataFrame]:
    config = _comparison_mode_config()
    if mode not in config:
        raise ValueError(f"Unknown comparison mode: {mode}")

    mode_cfg = config[mode]
    required_columns = [
        "usage category 1",
        "usage category 2",
        "metric value 1",
        "metric value 2",
        "model abbreviation 1",
        "model abbreviation 2",
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
    comparisons = comparisons.dropna(subset=required_columns).copy()
    comparisons = comparisons[
        comparisons["usage category 1"] == comparisons["usage category 2"]
    ].copy()

    if mode == "architecture":
        comparisons["node 1"] = comparisons["model architecture category 1"]
        comparisons["node 2"] = comparisons["model architecture category 2"]
    else:
        filter_categories = mode_cfg["filter_categories"]
        if (
            filter_categories is not None
            and "model architecture category 1" in comparisons.columns
            and "model architecture category 2" in comparisons.columns
        ):
            comparisons = comparisons[
                comparisons["model architecture category 1"].isin(filter_categories)
                & comparisons["model architecture category 2"].isin(filter_categories)
            ].copy()

        if "base model 1" in comparisons.columns and "base model 2" in comparisons.columns:
            comparisons = comparisons[
                comparisons["base model 1"].astype(str).str.strip().ne("")
                & comparisons["base model 2"].astype(str).str.strip().ne("")
            ].copy()
            comparisons["node 1"] = comparisons["base model 1"]
            comparisons["node 2"] = comparisons["base model 2"]
        else:
            comparisons["base model 1"] = comparisons.apply(
                lambda row: _base_model_label_for_row(
                    pd.Series({
                        "Category": row.get("model architecture category 1", ""),
                        "Type of model": row.get("Type of model 1", ""),
                        "Model abbreviation": row.get("model abbreviation 1", ""),
                    })
                ),
                axis=1,
            )
            comparisons["base model 2"] = comparisons.apply(
                lambda row: _base_model_label_for_row(
                    pd.Series({
                        "Category": row.get("model architecture category 2", ""),
                        "Type of model": row.get("Type of model 2", ""),
                        "Model abbreviation": row.get("model abbreviation 2", ""),
                    })
                ),
                axis=1,
            )
            comparisons = comparisons[
                comparisons["base model 1"].astype(str).str.strip().ne("")
                & comparisons["base model 2"].astype(str).str.strip().ne("")
            ].copy()
            comparisons["node 1"] = comparisons["base model 1"]
            comparisons["node 2"] = comparisons["base model 2"]

    comparisons = comparisons[comparisons["metric value 1"] != comparisons["metric value 2"]].copy()
    comparisons["winner"] = comparisons.apply(
        lambda row: row["node 1"]
        if row["metric value 1"] > row["metric value 2"]
        else row["node 2"],
        axis=1,
    )
    comparisons["loser"] = comparisons.apply(
        lambda row: row["node 2"]
        if row["metric value 1"] > row["metric value 2"]
        else row["node 1"],
        axis=1,
    )

    if not include_self_edges:
        comparisons = comparisons[comparisons["winner"] != comparisons["loser"]].copy()

    # Make the relevant node labels available for the graph builder.
    node_rows = comparisons.copy()
    nodes = sorted(
        set(node_rows["node 1"].dropna()) | set(node_rows["node 2"].dropna())
    )
    return nodes, comparisons




def create_model_architecture_win_graph(
    pairwise_df: pd.DataFrame,
    output_path: str | None = "model_architecture_win_graph.pdf",
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
                    color=PLOT_MUTED_COLOR,
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
                color=PLOT_MUTED_COLOR,
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
            node_color="#d9e9f5",
            edgecolors="#234",
            ax=ax,
        )
        for node in graph.nodes:
            x, y = pos[node]
            node_lines, font_size = _node_label_style(node, width=18)
            ax.text(
                x,
                y,
                "\n".join(node_lines),
                fontsize=font_size,
                fontweight="bold",
                ha="center",
                va="center",
                linespacing=0.90,
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


def _node_label_style(node: str, width: int = 20) -> tuple[list[str], float]:
    lines = _wrap_node_label(node, width=width)
    max_line_len = max((len(line) for line in lines), default=0)
    font_size = 8.4 - 0.14 * max(0, max_line_len - 12) - 0.25 * max(0, len(lines) - 1)
    font_size = max(6.0, min(8.2, font_size))
    return lines, font_size


def _normalize_docx_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text if text and text.lower() != "nan" else ""


def _normalize_doi(value) -> tuple[str, str, str]:
    """
    Return (display_text, href, title) for a study reference value.

    DOI strings are normalized to canonical doi.org links.
    Non-DOI URLs are kept as-is and labeled as generic study links.
    """
    text = _normalize_docx_text(value)
    if not text:
        return "", "", ""

    text = re.sub(r"\s+", "", text)
    lower = text.lower()
    doi_match = re.match(r"^(?:https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/\S+)$", text, flags=re.IGNORECASE)
    if doi_match:
        doi = doi_match.group(1)
        href = f"https://doi.org/{doi}"
        return "DOI", href, doi

    if lower.startswith("https://doi.org/") or lower.startswith("http://doi.org/"):
        doi = re.sub(r"^https?://doi\.org/", "", text, flags=re.IGNORECASE)
        href = f"https://doi.org/{doi}"
        return "DOI", href, doi

    if lower.startswith("https://dx.doi.org/") or lower.startswith("http://dx.doi.org/"):
        doi = re.sub(r"^https?://dx\.doi\.org/", "", text, flags=re.IGNORECASE)
        href = f"https://doi.org/{doi}"
        return "DOI", href, doi

    if lower.startswith(("https://", "http://")):
        parsed = urlparse(text)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        label_map = {
            "arxiv.org": "arXiv",
            "dx.doi.org": "DOI",
            "doi.org": "DOI",
            "github.com": "GitHub",
            "pubmed.ncbi.nlm.nih.gov": "PubMed",
            "proceedings.mlr.press": "PMLR",
            "pmlr.press": "PMLR",
            "zenodo.org": "Zenodo",
        }
        display = label_map.get(host, host or "Study link")
        return display, text, text

    return "DOI", f"https://doi.org/{text}", text


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

    parsed_eval = _unique_eval_sample_sizes_by_study(parse_eval_sample_sizes(df))
    parsed_eval_titles = set(
        _normalize_docx_text(value)
        for value in parsed_eval["Title"].dropna().tolist()
        if _normalize_docx_text(value)
    ) if "Title" in parsed_eval.columns else set()
    parsed_eval_by_title: dict[str, dict[str, list[float]]] = {}
    for _, parsed_row in parsed_eval.iterrows():
        title = _normalize_docx_text(parsed_row.get("Title", ""))
        unit = _normalize_docx_text(parsed_row.get("unit", "")).lower()
        value = pd.to_numeric(parsed_row.get("sample size", None), errors="coerce")
        if not title or pd.isna(value):
            continue
        bucket = parsed_eval_by_title.setdefault(title, {"texts": [], "patients": []})
        if unit == "texts":
            bucket["texts"].append(float(value))
        elif unit == "patients":
            bucket["patients"].append(float(value))

    rows = []
    for _, row in df.iterrows():
        year_value = pd.to_numeric(row["Year"], errors="coerce")
        shared = _normalize_docx_text(row["Model shared"])
        location = _normalize_docx_text(row["Code location(s)"]) if shared.lower() in {"yes", "partially"} else ""
        normalized_task_description = _normalize_docx_text(row["NLP Task description"])
        doi_display, doi_href, doi_title = _normalize_doi(row.get("DOI", ""))
        rows.append({
            "author": _normalize_docx_text(row["First author"]),
            "abbreviation": _normalize_docx_text(row["Model abbreviation"]),
            "title": _normalize_docx_text(row["Title"]),
            "year": int(year_value) if pd.notna(year_value) else "",
            "usage_category": _normalize_docx_text(row["Usage category"]),
            "nlp_task_description": normalized_task_description,
            "shared": shared,
            "model_location": location or "Not listed",
            "doi_display": doi_display,
            "doi_href": doi_href,
            "doi_title": doi_title,
            "evaluation_flag": _normalize_docx_text(row.get("Ev conducted yes/no", "")),
            "has_parsed_eval_sample_size": _normalize_docx_text(row["Title"]) in parsed_eval_titles,
        })

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
    parsed_eval_by_title_json = json.dumps(parsed_eval_by_title, ensure_ascii=False).replace("</", "<\\/")

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
    --blue: #2f6f9f;
    --blue-dark: #1f4f73;
    --blue-soft: #d9e9f5;
    --accent: #d46b35;
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
    grid-template-columns: repeat(5, minmax(120px, 1fr));
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
  .stat .value.multi {{
    font-size: 15px;
    line-height: 1.25;
    display: grid;
    gap: 2px;
  }}
  .stat .value.multi span {{
    display: block;
  }}
  .stat .subtle {{
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
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
  .doi-chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    max-width: 100%;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid rgba(17,145,250,.22);
    background: rgba(17,145,250,.08);
    color: var(--blue-dark);
    font-size: 11px;
    font-weight: 800;
    text-decoration: none;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .doi-chip:hover {{
    background: rgba(17,145,250,.14);
    border-color: rgba(17,145,250,.35);
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
  <div class="stat"><div class="label">Studies involved</div><div class="value" id="statStudies">0</div></div>
  <div class="stat"><div class="label">Models</div><div class="value" id="statModels">0</div></div>
  <div class="stat"><div class="label">Evaluations</div><div class="value" id="statEvaluations">0</div></div>
  <div class="stat"><div class="label">Shared</div><div class="value" id="statShared">0</div></div>
  <div class="stat">
    <div class="label">Eval sample size avg</div>
    <div class="value multi" id="statEvalSizes">
      <span>Texts: n/a</span>
      <span>Patients: n/a</span>
    </div>
  </div>
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
        <th>Study DOI / link</th>
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
const STUDY_EVAL_SIZES = __STUDY_EVAL_SIZES__;
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

function isEvaluated(value) {{
  return normalize(value) === "yes";
}}

function computeStats(rows) {{
  const studies = [...new Set(rows.map(row => row.title).filter(Boolean))].length;
  const titles = [...new Set(rows.map(row => row.title).filter(Boolean))];
  const textValues = [];
  const patientValues = [];
  const average = values => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  for (const title of titles) {{
    const study = STUDY_EVAL_SIZES[title];
    if (!study) continue;
    if (Array.isArray(study.texts) && study.texts.length) textValues.push(average(study.texts));
    if (Array.isArray(study.patients) && study.patients.length) patientValues.push(average(study.patients));
  }}
  return {{
    studies,
    models: new Set(rows.map(row => row.abbreviation).filter(Boolean)).size,
    evaluations: rows.filter(row => isEvaluated(row.evaluation_flag)).length,
    shared: rows.filter(row => isShared(row.shared)).length,
    evalSampleSizesTexts: {{
      count: textValues.length,
      mean: average(textValues),
    }},
    evalSampleSizesPatients: {{
      count: patientValues.length,
      mean: average(patientValues),
    }},
  }};
}}

function renderStats(rows) {{
  const stats = computeStats(rows);
  document.getElementById("statStudies").textContent = stats.studies.toLocaleString();
  document.getElementById("statModels").textContent = stats.models.toLocaleString();
  document.getElementById("statEvaluations").textContent = stats.evaluations.toLocaleString();
  document.getElementById("statShared").textContent = stats.shared.toLocaleString();
  const textMean = stats.evalSampleSizesTexts.mean === null ? "n/a" : stats.evalSampleSizesTexts.mean.toFixed(0);
  const patientMean = stats.evalSampleSizesPatients.mean === null ? "n/a" : stats.evalSampleSizesPatients.mean.toFixed(0);
  document.getElementById("statEvalSizes").innerHTML = `
    <span>Texts: ${{textMean}}${{stats.evalSampleSizesTexts.count ? ` <span class="subtle">(n=${{stats.evalSampleSizesTexts.count}})</span>` : ""}}</span>
    <span>Patients: ${{patientMean}}${{stats.evalSampleSizesPatients.count ? ` <span class="subtle">(n=${{stats.evalSampleSizesPatients.count}})</span>` : ""}}</span>
  `;
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
    row.doi_display,
    row.doi_title,
    row.usage_category,
    row.nlp_task_description,
    row.shared,
    row.model_location,
  ].some(value => normalize(value).includes(query));
}}

function render() {{
  const filtered = DATA.filter(passesFilters);
  renderStats(filtered);
  tableBody.innerHTML = filtered.map(row => `
    <tr>
      <td>${{esc(row.author)}}</td>
      <td>${{esc(row.abbreviation)}}</td>
      <td>${{esc(row.year)}}</td>
      <td>${{row.doi_href ? `<a class="doi-chip" href="${{esc(row.doi_href)}}" target="_blank" rel="noopener noreferrer" title="${{esc(row.doi_title || row.doi_href)}}">${{esc(row.doi_display || "DOI")}}</a>` : ""}}</td>
      <td>${{esc(row.usage_category)}}</td>
      <td>${{esc(row.nlp_task_description)}}</td>
      <td><span class="pill">${{esc(row.shared || "n/a")}}</span></td>
      <td>${{isShared(row.shared) ? esc(row.model_location) : ""}}</td>
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
        .replace("__STUDY_EVAL_SIZES__", parsed_eval_by_title_json)
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
        weight = edge.get("weight", edge.get("comparison_weight", edge.get("study_weight", 0)))
        pair_weights[pair] = pair_weights.get(pair, 0) + int(weight)

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
        output_path = output_dir_path / f"{_safe_filename(category)}.pdf"
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

    optional_columns = [
        col for col in ["Comparator block", "Title 1", "Title 2", "Main metric", "Ev metrics"]
        if col in pairwise_df.columns
    ]

    def _json_safe_scalar(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (pd.Timestamp,)):
            return str(value)
        return value

    graph_data = {}
    for mode_name, mode_cfg in _comparison_mode_config().items():
        _, comparisons = _prepare_comparison_graph_rows(
            pairwise_df,
            mode_name,
            include_self_edges=include_self_edges,
        )
        _, comparisons_with_self = _prepare_comparison_graph_rows(
            pairwise_df,
            mode_name,
            include_self_edges=True,
        )

        mode_graph = {
            "label": mode_cfg["label"],
            "detail_label": mode_cfg["detail_field"],
            "usage_categories": {},
        }
        if comparisons.empty:
            graph_data[mode_name] = mode_graph
            continue

        for usage_category, category_df in comparisons.groupby("usage category 1", sort=True):
            category_df_with_self = comparisons_with_self[
                comparisons_with_self["usage category 1"] == usage_category
            ].copy()
            graph_study_titles = set()
            node_arch_labels: dict[str, str] = {}
            node_arch_counts: dict[str, dict[str, int]] = {}
            for row_dict in category_df_with_self.to_dict(orient="records"):
                for title_key in ("Title 1", "Title 2", "Title"):
                    study_title = _normalize_docx_text(row_dict.get(title_key, ""))
                    if study_title:
                        graph_study_titles.add(study_title)
                for side in ("1", "2"):
                    node_label = _normalize_docx_text(row_dict.get(f"node {side}", ""))
                    arch_label = _normalize_docx_text(
                        row_dict.get(f"model architecture category {side}", "")
                    )
                    if node_label and arch_label:
                        node_arch_counts.setdefault(node_label, {})
                        node_arch_counts[node_label][arch_label] = (
                            node_arch_counts[node_label].get(arch_label, 0) + 1
                        )
            category_nodes = sorted(
                set(category_df["node 1"].dropna()) | set(category_df["node 2"].dropna())
            )
            for node_label, counts in node_arch_counts.items():
                node_arch_labels[node_label] = max(
                    counts.items(), key=lambda item: (item[1], item[0])
                )[0]
            if mode_name == "architecture":
                node_arch_labels.update({node: node for node in category_nodes})
            node_colors = _architecture_color_map(
                [node_arch_labels.get(node, "") for node in category_nodes]
            )
            def _build_edges(edge_source_df: pd.DataFrame) -> list[dict[str, object]]:
                edges_local = []
                grouped = edge_source_df.groupby(["winner", "loser"], sort=True)
                for (winner, loser), edge_df in grouped:
                    details = []
                    study_titles = set()
                    for row_dict in edge_df.to_dict(orient="records"):
                        winner_first = row_dict["metric value 1"] > row_dict["metric value 2"]
                        winner_side = "1" if winner_first else "2"
                        loser_side = "2" if winner_first else "1"
                        for title_key in ("Title 1", "Title 2", "Title"):
                            study_title = _normalize_docx_text(row_dict.get(title_key, ""))
                            if study_title:
                                study_titles.add(study_title)
                        detail = {
                            "winner_model": row_dict[f"model abbreviation {winner_side}"],
                            "winner_node": row_dict[f"node {winner_side}"],
                            "winner_metric_value": row_dict[f"metric value {winner_side}"],
                            "loser_model": row_dict[f"model abbreviation {loser_side}"],
                            "loser_node": row_dict[f"node {loser_side}"],
                            "loser_metric_value": row_dict[f"metric value {loser_side}"],
                            "winner_first_author": _json_safe_scalar(row_dict.get(f"First author {winner_side}", "")),
                            "winner_year": _json_safe_scalar(row_dict.get(f"Year {winner_side}", "")),
                            "winner_doi_display": _json_safe_scalar(row_dict.get(f"doi_display {winner_side}", "")),
                            "winner_doi_href": _json_safe_scalar(row_dict.get(f"doi_href {winner_side}", "")),
                            "winner_doi_title": _json_safe_scalar(row_dict.get(f"doi_title {winner_side}", "")),
                            "loser_first_author": _json_safe_scalar(row_dict.get(f"First author {loser_side}", "")),
                            "loser_year": _json_safe_scalar(row_dict.get(f"Year {loser_side}", "")),
                            "loser_doi_display": _json_safe_scalar(row_dict.get(f"doi_display {loser_side}", "")),
                            "loser_doi_href": _json_safe_scalar(row_dict.get(f"doi_href {loser_side}", "")),
                            "loser_doi_title": _json_safe_scalar(row_dict.get(f"doi_title {loser_side}", "")),
                        }
                        for field in ("Title", "First author", "Year"):
                            left_key = f"{field} 1"
                            right_key = f"{field} 2"
                            if left_key in row_dict:
                                detail[left_key] = _json_safe_scalar(row_dict.get(left_key, ""))
                            if right_key in row_dict:
                                detail[right_key] = _json_safe_scalar(row_dict.get(right_key, ""))
                        for field in ("doi_display", "doi_href", "doi_title"):
                            left_key = f"{field} 1"
                            right_key = f"{field} 2"
                            if left_key in row_dict:
                                detail[left_key] = _json_safe_scalar(row_dict.get(left_key, ""))
                            if right_key in row_dict:
                                detail[right_key] = _json_safe_scalar(row_dict.get(right_key, ""))
                        for col in optional_columns:
                            detail[col] = _json_safe_scalar(row_dict.get(col, ""))
                        details.append(detail)

                    comparison_weight = int(len(edge_df))
                    study_weight = int(len(study_titles)) if study_titles else comparison_weight
                    edges_local.append({
                        "source": winner,
                        "target": loser,
                        "comparison_weight": comparison_weight,
                        "study_weight": study_weight,
                        "details": details,
                    })
                return edges_local

            edges = _build_edges(category_df)
            edges_with_self = _build_edges(category_df_with_self)

            if not edges and not edges_with_self:
                continue

            edges = sorted(edges, key=lambda edge: edge["comparison_weight"], reverse=True)
            edges_with_self = sorted(edges_with_self, key=lambda edge: edge["comparison_weight"], reverse=True)
            mode_graph["usage_categories"][usage_category] = {
                "nodes": _weighted_circular_node_order(category_nodes, edges),
                "edges": edges,
                "edges_with_self": edges_with_self,
                "ranking_edges": edges,
                "ranking_edges_with_self": edges_with_self,
                "unique_study_count": int(len(graph_study_titles)),
                "node_colors": node_colors,
            }

        graph_data[mode_name] = mode_graph

    json_text = json.dumps(graph_data, ensure_ascii=False)
    json_text = json_text.replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Model Comparison Dashboard</title>
<style>
  :root {{
    --ink: #17212b;
    --muted: #5d6d7e;
    --line: #d6dde5;
    --blue: #2f6f9f;
    --blue-dark: #1f4f73;
    --blue-soft: #d9e9f5;
    --accent: #d46b35;
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
    grid-template-columns: minmax(220px, 280px) minmax(520px, 1fr) minmax(360px, 520px);
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
  .toolbar label {{
    color: var(--muted);
    font-size: 12px;
    white-space: nowrap;
  }}
  .checkbox-label {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    user-select: none;
  }}
  .checkbox-label input {{
    width: 16px;
    height: 16px;
    margin: 0;
  }}
  select {{
    max-width: 520px;
    padding: 8px 10px;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: #fff;
    color: var(--ink);
  }}
  .ranking {{
    padding: 14px;
    max-height: 770px;
    overflow: auto;
  }}
  .ranking h2 {{
    margin: 0 0 6px;
    font-size: 16px;
  }}
  .ranking .meta {{
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 12px;
  }}
  .ranking ol {{
    margin: 0;
    padding-left: 18px;
  }}
  .ranking li {{
    margin: 0 0 10px;
    line-height: 1.35;
  }}
  .ranking .node-name {{
    font-weight: 800;
  }}
  .ranking .score {{
    color: var(--muted);
    font-size: 11px;
    display: block;
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
    fill: #17212b;
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
    .ranking {{
      max-height: none;
    }}
    svg {{
      height: 620px;
    }}
  }}
</style>
</head>
<body>
<header>
  <h1>Model Comparison Dashboard</h1>
  <div class="hint">Switch between architecture-level and base-model-level views. Arrow direction is winner → loser.</div>
  <div class="refbar">This dashboard corresponds to the Zenodo preprint <a href="https://zenodo.org/records/19461436" target="_blank" rel="noopener">Natural language processing and language models for Dutch clinical text: a systematic review</a> (DOI 10.5281/zenodo.19461436).</div>
  <div class="nav">
    <a href="model_catalog_dashboard.html">Model catalog</a>
    <a href="model_architecture_win_graphs_interactive.html">Model comparison dashboard</a>
  </div>
</header>
<main class="layout">
  <aside class="panel">
    <div class="ranking" id="ranking">
      <h2>Elo ranking</h2>
      <div class="meta" id="rankingMeta">Select a usage category.</div>
      <div class="empty">The ranking will appear here.</div>
    </div>
  </aside>
  <section class="panel">
    <div class="toolbar">
      <label for="mode">Comparison level</label>
      <select id="mode"></select>
      <label for="weight">Arrow weight</label>
      <select id="weight"></select>
      <label class="checkbox-label" for="selfEdges">
        <input id="selfEdges" type="checkbox" />
        <span>Show self-directed edges</span>
      </label>
      <label for="category">Usage category</label>
      <select id="category"></select>
    </div>
    <svg id="graph" viewBox="0 0 900 720" role="img" aria-label="Directed model comparison graph"></svg>
  </section>
  <aside class="panel">
    <div id="details" class="empty">Select a usage category and click an arrow.</div>
  </aside>
</main>
<script id="graph-data" type="application/json">{json_text}</script>
<script>
const DATA = JSON.parse(document.getElementById("graph-data").textContent);
const modeSelect = document.getElementById("mode");
const weightSelect = document.getElementById("weight");
const selfEdgesSelect = document.getElementById("selfEdges");
const categorySelect = document.getElementById("category");
const ranking = document.getElementById("ranking");
const rankingMeta = document.getElementById("rankingMeta");
const svg = document.getElementById("graph");
const details = document.getElementById("details");
const NS = "http://www.w3.org/2000/svg";
let currentWeightMode = "comparison";

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

function paperReference(author, year, doiDisplay, doiHref, doiTitle) {{
  const citationParts = [];
  const authorText = (author ?? "").toString().trim();
  const yearText = (year ?? "").toString().trim();
  if (authorText) citationParts.push(esc(authorText));
  if (yearText) citationParts.push(esc(yearText));
  const citation = citationParts.join(", ");
  const doiChip = doiHref
    ? `<a class="doi-chip" href="${{esc(doiHref)}}" target="_blank" rel="noopener noreferrer" title="${{esc(doiTitle || doiHref)}}">${{esc(doiDisplay || "DOI")}}</a>`
    : "";
  const content = [citation, doiChip].filter(Boolean).join(" ");
  return content ? `[${{content}}]` : "";
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

function computeEloRatings(nodes, edges, getWeight) {{
  const labels = [...nodes];
  if (!labels.length) return [];
  if (labels.length === 1) {{
    return [{{node: labels[0], score: 1}}];
  }}

  const ratings = new Map(labels.map(node => [node, 1500]));
  const kFactor = 24;
  const iterations = 18;
  const shrinkage = 0.985;
  const epsilon = 1e-12;

  for (let iter = 0; iter < iterations; iter += 1) {{
    let maxDelta = 0;
    for (const edge of edges) {{
      const winner = edge.source;
      const loser = edge.target;
      if (winner === loser) continue;
      if (!ratings.has(winner) || !ratings.has(loser)) continue;
      const weight = Math.max(0, Number(getWeight(edge)) || 0);
      if (!weight) continue;
      const ratingWinner = ratings.get(winner);
      const ratingLoser = ratings.get(loser);
      const expectedWin = 1 / (1 + Math.pow(10, (ratingLoser - ratingWinner) / 400));
      const delta = kFactor * weight * (1 - expectedWin);
      const newWinner = ratingWinner + delta;
      const newLoser = ratingLoser - delta;
      maxDelta = Math.max(maxDelta, Math.abs(newWinner - ratingWinner), Math.abs(newLoser - ratingLoser));
      ratings.set(winner, newWinner);
      ratings.set(loser, newLoser);
    }}

    const mean = [...ratings.values()].reduce((sum, value) => sum + value, 0) / labels.length;
    for (const node of labels) {{
      const centered = (ratings.get(node) || 1500) + (1500 - mean);
      ratings.set(node, 1500 + (centered - 1500) * shrinkage);
    }}
    if (maxDelta < epsilon) break;
  }}

  return labels
    .map(node => ({{node, score: ratings.get(node) || 1500}}))
    .sort((a, b) => {{
      if (b.score !== a.score) return b.score - a.score;
      return a.node.localeCompare(b.node, undefined, {{numeric: true, sensitivity: "base"}});
    }});
}}

function renderRanking(mode, category, data, getWeight, weightMode, includeSelfEdges) {{
  const rankingEdges = includeSelfEdges
    ? (data.ranking_edges_with_self || data.ranking_edges || data.edges_with_self || data.edges || [])
    : (data.ranking_edges || data.edges || []);
  const entries = computeEloRatings(data.nodes || [], rankingEdges, getWeight);
  const weightLabel = weightMode === "study" ? "studies" : "model comparisons";
  const titleLabel = (DATA[mode] && DATA[mode].label) ? DATA[mode].label : mode;
  const studyCount = Number(data.unique_study_count || 0);
  const studyLabel = studyCount ? `${{studyCount}} unique studies` : "no study count available";
  const metaText = `${{titleLabel}} · ${{category || "All usage categories"}} · ${{studyLabel}} · weighted by ${{weightLabel}} · full win/loss set`;
  if (!entries.length) {{
    ranking.innerHTML = `
      <h2>Elo ranking</h2>
      <div class="meta">${{esc(metaText)}}</div>
      <div class="empty">No nodes available for this graph.</div>
    `;
    return;
  }}

  const maxScore = Math.max(...entries.map(entry => entry.score), 0) || 1;
  ranking.innerHTML = `
    <h2>Elo ranking</h2>
    <div class="meta">${{esc(metaText)}}</div>
    <ol>
      ${{entries.map((entry, idx) => `
        <li>
          <span class="node-name">${{idx + 1}}. ${{esc(entry.node)}}</span>
          <span class="score">Elo: ${{entry.score.toFixed(1)}} · rel. ${{(entry.score / maxScore * 100).toFixed(1)}}%</span>
        </li>
      `).join("")}}
    </ol>
  `;
}}

let currentMode = "";

function renderDetails(edge, weightMode) {{
  const weightLabel = weightMode === "study" ? "studies" : "model comparisons";
  details.className = "";
  details.innerHTML = `
    <div class="detail-title">${{esc(edge.source)}} wins over ${{esc(edge.target)}}</div>
    <div class="detail-sub">Weighted by ${{weightLabel}}. ${{edge.comparison_weight}} model comparison(s) across ${{edge.study_weight}} study(s). Arrow points at the loser.</div>
    <table>
      <thead>
        <tr>
          <th>Winning model</th>
          <th>Win value</th>
          <th>Losing model</th>
          <th>Loss value</th>
          <th>Metric</th>
          <th>Comparator block</th>
        </tr>
      </thead>
      <tbody>
        ${{edge.details.map(d => `
          <tr>
            <td>
              ${{esc(d.winner_model)}}
              ${{paperReference(d.winner_first_author, d.winner_year, d.winner_doi_display, d.winner_doi_href, d.winner_doi_title)}}
              <br/><small>${{esc(d.winner_node)}}</small>
            </td>
            <td>${{numberText(d.winner_metric_value)}}</td>
            <td>
              ${{esc(d.loser_model)}}
              ${{paperReference(d.loser_first_author, d.loser_year, d.loser_doi_display, d.loser_doi_href, d.loser_doi_title)}}
              <br/><small>${{esc(d.loser_node)}}</small>
            </td>
            <td>${{numberText(d.loser_metric_value)}}</td>
            <td>${{esc(d["Main metric"] || d["Ev metrics"] || "")}}</td>
            <td>${{esc(d["Comparator block"] || "")}}</td>
          </tr>
        `).join("")}}
      </tbody>
    </table>
  `;
}}

function renderModeOptions() {{
  const entries = Object.entries(DATA);
  modeSelect.innerHTML = entries.map(([key, value]) => `<option value="${{esc(key)}}">${{esc(value.label || key)}}</option>`).join("");
}}

function renderWeightOptions() {{
  weightSelect.innerHTML = `
    <option value="comparison">Model comparisons</option>
    <option value="study">Studies</option>
  `;
}}

function renderCategoryOptions(mode) {{
  const modeData = DATA[mode] || {{}};
  const categories = Object.keys(modeData.usage_categories || {{}}).sort();
  categorySelect.innerHTML = categories.map(value => `<option value="${{esc(value)}}">${{esc(value)}}</option>`).join("");
}}

function renderGraph(mode, category, weightMode = "comparison", includeSelfEdges = false) {{
  currentWeightMode = weightMode;
  svg.innerHTML = "";
  svg.setAttribute("viewBox", "0 0 980 760");
  details.className = "empty";
  details.textContent = "Click an arrow to inspect the comparisons behind it.";

  const modeData = DATA[mode] || {{usage_categories: {{}}}};
  const data = modeData.usage_categories?.[category] || {{nodes: [], edges: []}};
  const baseEdges = includeSelfEdges ? (data.edges_with_self || data.edges || []) : (data.edges || []);
  const width = 980;
  const height = 760;
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(width, height) * 0.44;
  const nodeCount = data.nodes.length || 1;
  const nodeRadius = Math.max(
    18,
    Math.min(44, Math.round(46 - (nodeCount * 0.65)))
  );
  const getWeight = (edge) => weightMode === "study" ? edge.study_weight : edge.comparison_weight;
  const weights = baseEdges.map(e => getWeight(e));
  const minLog = weights.length ? Math.min(...weights.map(w => Math.log1p(w))) : 0;
  const maxLog = weights.length ? Math.max(...weights.map(w => Math.log1p(w))) : 0;
  const strength = (w) => maxLog === minLog ? 1 : (Math.log1p(w) - minLog) / (maxLog - minLog);
  const edgeRank = new Map(
    [...baseEdges]
      .sort((a, b) => getWeight(b) - getWeight(a))
      .map((edge, idx) => [`${{edge.source}}|||${{edge.target}}`, idx])
  );
  const renderedEdges = [...baseEdges].sort((a, b) => {{
    const weightA = getWeight(a);
    const weightB = getWeight(b);
    if (weightA !== weightB) return weightA - weightB;
    const sourceCmp = (a.source || "").localeCompare(b.source || "", undefined, {{numeric: true, sensitivity: "base"}});
    if (sourceCmp !== 0) return sourceCmp;
    return (a.target || "").localeCompare(b.target || "", undefined, {{numeric: true, sensitivity: "base"}});
  }});
  const isPromptedBase = mode === "prompted_base";

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
  marker.appendChild(makeSvg("path", {{d: "M0,0 L0,6 L9,3 z", fill: "#2f6f9f"}}));
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
  hoverMarkerVisible.appendChild(makeSvg("path", {{d: "M0,0 L0,6 L9,3 z", fill: "#d46b35"}}));
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

  renderedEdges.forEach((edge, i) => {{
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) return;
    const edgeWeight = getWeight(edge);
    const s = strength(edgeWeight);
    const strokeWidth = 1 + 5 * s;
    const edgeKey = `${{edge.source}}|||${{edge.target}}`;
    const rank = edgeRank.get(edgeKey) ?? i;
    const opacity = isPromptedBase
      ? (rank < 10 ? 0.96 - (rank * 0.06) : 0.05)
      : (0.18 + 0.72 * s);
    const rad = 0.22;
    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const length = Math.hypot(dx, dy) || 1;
    let path = "";
    let label = {{x: source.x, y: source.y, angle: 0}};

    if (edge.source === edge.target) {{
      const centerDx = source.x - cx;
      const centerDy = source.y - cy;
      const centerLength = Math.hypot(centerDx, centerDy) || 1;
      const outX = centerDx / centerLength;
      const outY = centerDy / centerLength;
      const sideX = -outY;
      const sideY = outX;
      const loopRadius = nodeRadius * 2.4 + 22 + (18 * s);
      const spread = nodeRadius * 1.15;
      const start = {{
        x: source.x + outX * (nodeRadius * 1.05) + sideX * spread * 0.35,
        y: source.y + outY * (nodeRadius * 1.05) + sideY * spread * 0.35,
      }};
      const end = {{
        x: source.x + outX * (nodeRadius * 1.05) - sideX * spread * 0.35,
        y: source.y + outY * (nodeRadius * 1.05) - sideY * spread * 0.35,
      }};
      const control1 = {{
        x: source.x + outX * loopRadius + sideX * loopRadius * 0.72,
        y: source.y + outY * loopRadius + sideY * loopRadius * 0.72,
      }};
      const control2 = {{
        x: source.x + outX * loopRadius - sideX * loopRadius * 0.72,
        y: source.y + outY * loopRadius - sideY * loopRadius * 0.72,
      }};
      path = `M ${{start.x}} ${{start.y}} C ${{control1.x}} ${{control1.y}} ${{control2.x}} ${{control2.y}} ${{end.x}} ${{end.y}}`;
      label = {{
        x: source.x + outX * (loopRadius * 0.95) + sideX * (loopRadius * 0.55),
        y: source.y + outY * (loopRadius * 0.95) + sideY * (loopRadius * 0.55),
        angle: 0,
      }};
    }} else {{
      const ux = dx / length;
      const uy = dy / length;
      const start = {{x: source.x + ux * nodeRadius, y: source.y + uy * nodeRadius}};
      const end = {{x: target.x - ux * nodeRadius, y: target.y - uy * nodeRadius}};
      const normal = {{x: -uy, y: ux}};
      const control = {{
        x: (start.x + end.x) / 2 + normal.x * rad * length,
        y: (start.y + end.y) / 2 + normal.y * rad * length,
      }};
      path = `M ${{start.x}} ${{start.y}} Q ${{control.x}} ${{control.y}} ${{end.x}} ${{end.y}}`;
      label = curvePoint(start, end, rad);
    }}

    const group = makeSvg("g", {{"class": "edge"}});
    group.appendChild(makeSvg("path", {{
      class: "hit-area",
      d: path,
      "marker-end": "url(#arrowhead-hit-area)",
      "stroke-width": edge.source === edge.target ? Math.max(32, strokeWidth * 4) : 22,
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
    }}, edgeWeight.toString()));
    group.addEventListener("click", () => {{
      svg.querySelectorAll(".edge").forEach(el => el.classList.remove("active"));
      group.classList.add("active");
      currentMode = mode;
      renderDetails(edge, weightMode);
    }});
    svg.appendChild(group);
  }});

  renderRanking(mode, category, data, getWeight, weightMode, includeSelfEdges);

  data.nodes.forEach(node => {{
    const p = positions.get(node);
    const group = makeSvg("g", {{"class": "node"}});
    group.appendChild(makeSvg("circle", {{
      cx: p.x,
      cy: p.y,
      r: nodeRadius,
      fill: data.node_colors?.[node] || "#d9e9f5",
    }}));
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

renderModeOptions();
renderWeightOptions();
currentMode = modeSelect.options.length ? modeSelect.value : "";
currentWeightMode = weightSelect.options.length ? weightSelect.value : currentWeightMode;
renderCategoryOptions(currentMode);

modeSelect.addEventListener("change", () => {{
  currentMode = modeSelect.value;
  renderCategoryOptions(currentMode);
  categorySelect.selectedIndex = 0;
  renderGraph(currentMode, categorySelect.value, weightSelect.value || currentWeightMode, selfEdgesSelect.checked);
}});

weightSelect.addEventListener("change", () => {{
  currentWeightMode = weightSelect.value;
  renderGraph(currentMode, categorySelect.value, currentWeightMode, selfEdgesSelect.checked);
}});

selfEdgesSelect.addEventListener("change", () => renderGraph(currentMode, categorySelect.value, weightSelect.value || currentWeightMode, selfEdgesSelect.checked));

categorySelect.addEventListener("change", () => renderGraph(currentMode, categorySelect.value, weightSelect.value || currentWeightMode, selfEdgesSelect.checked));

if (modeSelect.options.length) {{
  modeSelect.selectedIndex = 0;
  currentMode = modeSelect.value;
  if (weightSelect.options.length) {{
    weightSelect.selectedIndex = 0;
    currentWeightMode = weightSelect.value;
  }}
  renderCategoryOptions(currentMode);
  if (categorySelect.options.length) {{
    categorySelect.selectedIndex = 0;
  }}
  renderGraph(currentMode, categorySelect.value, weightSelect.value || currentWeightMode, selfEdgesSelect.checked);
}} else {{
  details.textContent = "No comparisons available.";
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
        [
            "Dev region",
            "Ev region",
            # Study-level note and region fields can legitimately contain
            # multiple text types or multiple regions for a single study.
            "Dev text type",
            "Ev text type",
            "NLP Task description",
            "Dev size",
            "Ev size",
            "Ev metrics",
            "Contextual qualifier(s)"
        ],
        "raw_data_mappings",
    )

    pairwise = create_comparator_graph(df)
    graph, edge_weights = create_model_architecture_win_graph(pairwise)
    category_graphs = create_model_architecture_win_graphs_by_usage_category(pairwise)
    interactive_html = create_interactive_model_architecture_graph_html(pairwise)
    comparison_counts = _architecture_comparison_counts_by_usage_category(pairwise)
    parsed_eval_sizes = plot_eval_sample_size_distributions(df)
    text_type_distributions = plot_text_type_study_distributions(df)
    region_province_maps = plot_region_study_province_maps(df)
    any_text_region_panel = plot_any_dev_eval_text_region_panel(df)
    ev_metric_overview = plot_ev_metric_overview_by_usage_category(df)
    contextual_qualifier_overview = plot_contextual_qualifier_matrix_by_usage_category(df)
    metrics_sample_size_panel = plot_evaluation_metrics_sample_size_panel(df)
    publication_panel = plot_metrics_eval_sizes_ie_panel(df, pairwise)
    architecture_year_plots = plot_model_architecture_percentages_by_year(df)
    model_catalog_html = create_model_catalog_dashboard_html(df)
    available_models_docx = export_available_models_docx(df)
    architecture_study_counts = (
        df.dropna(subset=["Title", "Model architecture category"])
        .drop_duplicates(["Title", "Model architecture category"])
        .groupby("Model architecture category", as_index=False)
        .size()
        .sort_values(["size", "Model architecture category"], ascending=[False, True])
    )
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
    print("\nNumber of studies using each architecture type:")
    print(architecture_study_counts.to_string(index=False))
    print("Wrote matching study-count plots to model_architecture_percentages_by_year/.")
    print("Wrote panel study-count figure to model_architecture_percentages_by_year/architecture_study_counts_by_year_panels.pdf.")
    print(f"Wrote graph visualization with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(
        "Wrote separate usage-category graph visualizations to "
        "model_architecture_win_graphs_by_usage_category/ "
        f"({len(category_graphs)} files)."
    )
    print(f"Wrote interactive graph HTML to {interactive_html}.")
    print(f"Wrote model catalog HTML to {model_catalog_html}.")
    print("Wrote evaluation sample-size distributions to eval_sample_size_distributions/.")
    print(f"Wrote any-dev/eval text+region panel to {any_text_region_panel}.")
    print(f"Wrote evaluation metric overview to {ev_metric_overview['heatmap_pdf']}.")
    print(f"Wrote contextual qualifier matrix to {contextual_qualifier_overview['matrix_pdf']}.")
    print(f"Wrote evaluation metrics/sample-size panel to {metrics_sample_size_panel['pdf']}.")
    print(f"Wrote publication panel to {publication_panel['pdf']}.")
    print("Wrote architecture-by-year distributions to model_architecture_percentages_by_year/.")
    print(f"Wrote available-model table to {available_models_docx}.")
