"""Interactive four-model Office-Home demonstration.

The app deliberately performs real CPU inference one configuration at a time.
It never substitutes archived metrics for an individual prediction.
"""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import altair as alt
import pandas as pd
import streamlit as st

try:
    from demo.inference import (
        DOMAIN_ORDER,
        CheckpointError,
        CheckpointSpec,
        SampleError,
        SampleSpec,
        ensure_sample,
        load_checkpoint_specs,
        load_image,
        load_sample_specs,
        predict_images_for_configuration,
    )
except ModuleNotFoundError:  # Supports `streamlit run demo/app.py`.
    from inference import (
        DOMAIN_ORDER,
        CheckpointError,
        CheckpointSpec,
        SampleError,
        SampleSpec,
        ensure_sample,
        load_checkpoint_specs,
        load_image,
        load_sample_specs,
        predict_images_for_configuration,
    )


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "results_all.csv"
CLASS_MAPPING_PATH = ROOT / "results" / "class_to_idx.json"
CHECKPOINT_MANIFEST_PATH = ROOT / "demo" / "checkpoints.json"
SAMPLE_MANIFEST_PATH = ROOT / "demo" / "sample_manifest.json"

DOMAIN_LABELS = {
    "Real_World": "Real-World",
    "Product": "Product",
    "Art": "Art",
    "Clipart": "Clipart",
}
CONFIGURATION_ORDER = (
    "resnet18_linear_probe",
    "resnet18_finetune_last_block",
    "resnet50_linear_probe",
    "resnet50_finetune_last_block",
)
CHALLENGE_CHOICES = {
    "All four configurations": CONFIGURATION_ORDER,
    "ResNet18 vs ResNet50 — Linear Probe": (
        "resnet18_linear_probe",
        "resnet50_linear_probe",
    ),
    "ResNet18 vs ResNet50 — Fine-tune": (
        "resnet18_finetune_last_block",
        "resnet50_finetune_last_block",
    ),
    "ResNet18 — Linear Probe vs Fine-tune": (
        "resnet18_linear_probe",
        "resnet18_finetune_last_block",
    ),
    "ResNet50 — Linear Probe vs Fine-tune": (
        "resnet50_linear_probe",
        "resnet50_finetune_last_block",
    ),
}


st.set_page_config(
    page_title="Office-Home model demo",
    page_icon=":material/model_training:",
    layout="wide",
)


def pretty_class(class_name: str) -> str:
    return class_name.replace("_", " ")


def configuration_label(spec: CheckpointSpec) -> str:
    return spec.label


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    """Load saved measurements only; this app never recomputes experiment metrics."""
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Missing archived results: {RESULTS_PATH}")
    results = pd.read_csv(RESULTS_PATH)
    required = {
        "model",
        "mode",
        "test_domain",
        "accuracy",
        "macro_f1",
        "accuracy_drop_pp",
        "best_epoch",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"The archived results are missing: {', '.join(sorted(missing))}")
    return results


@st.cache_data(show_spinner=False)
def load_class_names() -> list[str]:
    mapping = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
    classes = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    if len(classes) != 65:
        raise ValueError("results/class_to_idx.json must contain 65 classes.")
    return classes


@st.cache_data(show_spinner=False)
def cached_checkpoint_specs() -> dict[str, CheckpointSpec]:
    return load_checkpoint_specs(CHECKPOINT_MANIFEST_PATH)


@st.cache_data(show_spinner=False)
def cached_sample_specs() -> dict[str, dict[str, SampleSpec]]:
    return load_sample_specs(SAMPLE_MANIFEST_PATH)


def initialise_state() -> None:
    st.session_state.setdefault("playground_run", None)
    st.session_state.setdefault("comparison_run", None)
    st.session_state.setdefault("challenge_run", None)


def result_key(image_fingerprint: str, configuration_keys: Sequence[str]) -> str:
    joined = "|".join(configuration_keys)
    return hashlib.sha256(f"{image_fingerprint}|{joined}".encode("utf-8")).hexdigest()


def selected_configuration_rows(results: pd.DataFrame, spec: CheckpointSpec) -> pd.DataFrame:
    rows = results[(results["model"] == spec.architecture) & (results["mode"] == spec.mode)].copy()
    rows = rows.set_index("test_domain").reindex(DOMAIN_ORDER).reset_index()
    if rows["accuracy"].isna().any():
        raise ValueError(f"Saved results are incomplete for {spec.label}.")
    rows["Domain"] = rows["test_domain"].map(DOMAIN_LABELS)
    rows["Accuracy (%)"] = rows["accuracy"] * 100
    rows["Macro-F1"] = rows["macro_f1"]
    rows["Drop from Real-World (pp)"] = rows["accuracy_drop_pp"]
    return rows


def configuration_table(
    results: pd.DataFrame,
    specs: Mapping[str, CheckpointSpec],
    metric: str,
) -> pd.DataFrame:
    data = results.copy()
    data["Configuration"] = data.apply(
        lambda row: specs[f"{row['model']}_{row['mode']}"].label,
        axis=1,
    )
    data["Domain"] = data["test_domain"].map(DOMAIN_LABELS)
    ordered_labels = [specs[key].label for key in CONFIGURATION_ORDER]
    table = data.pivot(index="Configuration", columns="Domain", values=metric)
    return table.reindex(index=ordered_labels, columns=[DOMAIN_LABELS[domain] for domain in DOMAIN_ORDER])


def render_header() -> None:
    st.title("Pretrained Vision Models Across Image Domains")
    st.caption("A real CPU inference demo for the completed Office-Home experiment.")
    st.markdown(
        """```text
ImageNet-pretrained
        |
  +-----+-----+
  |           |
ResNet18    ResNet50
  |           |
Linear / Fine-tune Last Block
        |
      4 domains
Real-World  Product  Art  Clipart
```"""
    )


def load_prepared_sample(
    sample: SampleSpec,
) -> tuple[Any | None, str | None]:
    try:
        return load_image(ensure_sample(sample)), None
    except SampleError as error:
        return None, str(error)


def open_uploaded_image(uploaded_file: Any) -> tuple[Any | None, str | None, str | None]:
    if uploaded_file is None:
        return None, None, None
    try:
        from PIL import Image

        raw = uploaded_file.getvalue()
        with Image.open(BytesIO(raw)) as uploaded:
            image = uploaded.convert("RGB").copy()
        return image, hashlib.sha256(raw).hexdigest(), None
    except Exception as error:  # Pillow exposes several decoder-specific exceptions.
        return None, None, f"The uploaded file could not be read as an image: {error}"


def image_picker(
    prefix: str,
    samples: Mapping[str, Mapping[str, SampleSpec]],
) -> tuple[Any | None, dict[str, str] | None, str | None]:
    source = st.segmented_control(
        "Image source",
        ["Prepared Office-Home sample", "Upload a JPG or PNG"],
        default="Prepared Office-Home sample",
        required=True,
        key=f"{prefix}_source",
        width="stretch",
    )
    if source == "Prepared Office-Home sample":
        classes = list(samples)
        selected_class = st.selectbox(
            "Prepared class",
            classes,
            key=f"{prefix}_class",
            format_func=pretty_class,
        )
        selected_domain = st.selectbox(
            "Prepared domain",
            list(DOMAIN_ORDER),
            key=f"{prefix}_domain",
            format_func=lambda domain: DOMAIN_LABELS[domain],
        )
        sample = samples[selected_class][selected_domain]
        with st.spinner("Resolving the selected Office-Home sample..."):
            image, error = load_prepared_sample(sample)
        if error:
            st.error(error, icon=":material/error:")
            return None, None, None
        metadata = {
            "class_name": sample.class_name,
            "domain": sample.domain,
            "split": sample.split,
            "relative_path": sample.relative_path,
        }
        fingerprint = f"sample:{sample.relative_path}:{sample.row_index}"
        return image, metadata, fingerprint

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key=f"{prefix}_upload",
    )
    image, fingerprint, error = open_uploaded_image(uploaded)
    if error:
        st.error(error, icon=":material/error:")
    if image is None:
        st.info("Choose a JPG, JPEG, or PNG image to enable inference.", icon=":material/upload:")
    return image, None, fingerprint


def render_image_context(image: Any, metadata: Mapping[str, str] | None, caption: str) -> None:
    st.image(image, caption=caption, width="stretch")
    if metadata is not None:
        st.caption(
            f"Known class: {pretty_class(metadata['class_name'])} · "
            f"Domain: {DOMAIN_LABELS[metadata['domain']]} · "
            f"Archived split: {metadata['split']}"
        )


def run_configurations(
    specs: Mapping[str, CheckpointSpec],
    configuration_keys: Sequence[str],
    images: Mapping[str, Any],
    class_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Run selected configurations sequentially, retaining probabilities but not models."""
    outcomes: dict[str, dict[str, Any]] = {}
    total = len(configuration_keys)
    progress = st.progress(0.0, text="Preparing checkpoint...")
    with st.status("Preparing checkpoint...", expanded=True) as status:
        for index, key in enumerate(configuration_keys, start=1):
            spec = specs[key]
            status.update(label=f"Running {spec.label}...")
            status.write(f"Running {spec.label} on CPU.")

            def checkpoint_progress(completed: int, size: int, position: int = index) -> None:
                fraction = completed / size if size else 0.0
                progress.progress(
                    min(1.0, ((position - 1) + fraction) / total),
                    text=f"Preparing checkpoint: {spec.label}",
                )

            try:
                predictions = predict_images_for_configuration(
                    spec,
                    images,
                    class_names,
                    progress_callback=checkpoint_progress,
                )
                outcomes[key] = {"predictions": predictions}
                status.write(f"Completed {spec.label}.")
            except (CheckpointError, RuntimeError, OSError) as error:
                outcomes[key] = {"error": str(error)}
                status.write(f"{spec.label} failed: {error}")
            progress.progress(index / total, text=f"Completed {spec.label}")
        status.update(label="Selected model runs complete", state="complete", expanded=False)
    progress.empty()
    return outcomes


def prediction_card(
    spec: CheckpointSpec,
    outcome: Mapping[str, Any] | None,
    image_name: str,
    expected_class: str | None,
) -> None:
    with st.container(border=True):
        st.subheader(spec.label)
        if not outcome:
            st.caption("No run recorded yet.")
            return
        if "error" in outcome:
            st.error(str(outcome["error"]), icon=":material/error:")
            return
        top_five = outcome["predictions"][image_name]
        top_one = top_five[0]
        st.metric("Top-1 prediction", pretty_class(str(top_one["class"])), border=True)
        st.caption(f"Top-1 confidence: {float(top_one['probability']):.1%}")
        if expected_class is not None:
            if top_one["class"] == expected_class:
                st.success("Correct for this labeled sample.", icon=":material/check_circle:")
            else:
                st.warning(
                    f"Incorrect for this labeled sample. Expected {pretty_class(expected_class)}.",
                    icon=":material/warning:",
                )
        top_table = pd.DataFrame(top_five)
        top_table["Probability (%)"] = top_table.pop("probability") * 100
        top_table["Class"] = top_table.pop("class").map(pretty_class)
        st.dataframe(
            top_table[["Class", "Probability (%)"]],
            hide_index=True,
            height=220,
            column_config={"Probability (%)": st.column_config.NumberColumn(format="%.2f%%")},
        )


def render_prediction_grid(
    specs: Mapping[str, CheckpointSpec],
    outcomes: Mapping[str, Mapping[str, Any]],
    configuration_keys: Sequence[str],
    image_name: str,
    expected_class: str | None,
) -> None:
    for start in range(0, len(configuration_keys), 2):
        columns = st.columns(2)
        for column, key in zip(columns, configuration_keys[start : start + 2], strict=False):
            with column:
                prediction_card(specs[key], outcomes.get(key), image_name, expected_class)


def valid_top_one(outcomes: Mapping[str, Mapping[str, Any]], key: str, image_name: str) -> dict[str, Any] | None:
    outcome = outcomes.get(key)
    if not outcome or "error" in outcome:
        return None
    predictions = outcome.get("predictions", {}).get(image_name)
    return predictions[0] if predictions else None


def render_pair_comparison(
    title: str,
    left_label: str,
    left: dict[str, Any] | None,
    right_label: str,
    right: dict[str, Any] | None,
    left_top_five: Sequence[Mapping[str, Any]] | None = None,
    right_top_five: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    with st.container(border=True):
        st.subheader(title)
        if left is None or right is None:
            st.warning("This comparison needs successful predictions from both configurations.")
            return
        difference = (float(right["probability"]) - float(left["probability"])) * 100
        first, second = st.columns(2)
        with first:
            st.metric(left_label, pretty_class(str(left["class"])), border=True)
            st.caption(f"Confidence: {float(left['probability']):.1%}")
        with second:
            st.metric(right_label, pretty_class(str(right["class"])), border=True)
            st.caption(f"Confidence: {float(right['probability']):.1%}")
        st.caption(f"Right minus left confidence: {difference:+.2f} percentage points on this image.")
        if left["class"] != right["class"]:
            st.warning(
                f"Top-1 prediction changed: {pretty_class(str(left['class']))} → {pretty_class(str(right['class']))}.",
                icon=":material/swap_horiz:",
            )
        else:
            st.success("Both configurations chose the same top-1 class.", icon=":material/check_circle:")
        if left_top_five is not None and right_top_five is not None:
            overlap = sorted(
                {str(row["class"]) for row in left_top_five}.intersection(
                    {str(row["class"]) for row in right_top_five}
                )
            )
            readable = ", ".join(pretty_class(item) for item in overlap) or "None"
            st.caption(f"Top-5 overlap ({len(overlap)}/5): {readable}")


def top_five_for(outcomes: Mapping[str, Mapping[str, Any]], key: str, image_name: str) -> Sequence[Mapping[str, Any]] | None:
    outcome = outcomes.get(key)
    if not outcome or "error" in outcome:
        return None
    return outcome.get("predictions", {}).get(image_name)


def render_playground(
    specs: Mapping[str, CheckpointSpec],
    samples: Mapping[str, Mapping[str, SampleSpec]],
    class_names: Sequence[str],
) -> None:
    st.header("Playground")
    st.caption("Choose one prepared Office-Home image or upload your own, then run all four trained configurations.")
    image, metadata, fingerprint = image_picker("playground", samples)
    if image is None or fingerprint is None:
        return

    preview, action = st.columns((1, 1), gap="large")
    with preview:
        render_image_context(image, metadata, "Selected image")
    with action:
        st.markdown("**Archived test preprocessing**")
        st.caption("RGB → resize shorter side to 256 → center crop 224 → ImageNet normalization.")
        st.caption("Each model is loaded on CPU only when this button is pressed, then released before the next model runs.")
        run = st.button(
            "Run all 4 models",
            type="primary",
            icon=":material/play_arrow:",
            key="playground_run_button",
            width="stretch",
        )

    run_id = result_key(fingerprint, CONFIGURATION_ORDER)
    if run:
        st.session_state["playground_run"] = {
            "run_id": run_id,
            "outcomes": run_configurations(
                specs,
                CONFIGURATION_ORDER,
                {"selected": image},
                class_names,
            ),
        }
    stored = st.session_state.get("playground_run")
    if not stored or stored.get("run_id") != run_id:
        return

    st.subheader("Four-model predictions")
    expected_class = metadata["class_name"] if metadata is not None else None
    render_prediction_grid(specs, stored["outcomes"], CONFIGURATION_ORDER, "selected", expected_class)
    st.subheader("Comparison summary")
    outcomes = stored["outcomes"]
    successful = [
        (key, valid_top_one(outcomes, key, "selected"))
        for key in CONFIGURATION_ORDER
    ]
    successful = [(key, value) for key, value in successful if value is not None]
    if successful:
        best_key, best = max(successful, key=lambda item: float(item[1]["probability"]))
        st.info(
            f"Highest top-1 confidence on this image: **{specs[best_key].label}** "
            f"({float(best['probability']):.1%}, {pretty_class(str(best['class']))}).",
            icon=":material/insights:",
        )
    architecture_left = valid_top_one(outcomes, "resnet18_linear_probe", "selected")
    architecture_right = valid_top_one(outcomes, "resnet50_linear_probe", "selected")
    strategy_left = valid_top_one(outcomes, "resnet50_linear_probe", "selected")
    strategy_right = valid_top_one(outcomes, "resnet50_finetune_last_block", "selected")
    summary_left, summary_right = st.columns(2)
    with summary_left:
        render_pair_comparison(
            "Architecture difference — linear probe",
            "ResNet18",
            architecture_left,
            "ResNet50",
            architecture_right,
            top_five_for(outcomes, "resnet18_linear_probe", "selected"),
            top_five_for(outcomes, "resnet50_linear_probe", "selected"),
        )
    with summary_right:
        render_pair_comparison(
            "Fine-tuning difference — ResNet50",
            "Linear Probe",
            strategy_left,
            "Fine-tune",
            strategy_right,
            top_five_for(outcomes, "resnet50_linear_probe", "selected"),
            top_five_for(outcomes, "resnet50_finetune_last_block", "selected"),
        )
    st.caption("Confidence is a model probability for this image. It is not an accuracy estimate or a measure of general model quality.")


def render_domain_shift_challenge(
    specs: Mapping[str, CheckpointSpec],
    samples: Mapping[str, Mapping[str, SampleSpec]],
    class_names: Sequence[str],
    results: pd.DataFrame,
) -> None:
    st.header("Domain Shift Challenge")
    st.caption("Compare the same labeled Office-Home class across four visual domains.")
    selected_class = st.selectbox(
        "Class",
        list(samples),
        format_func=pretty_class,
        key="challenge_class",
    )
    choice = st.selectbox("Compare", list(CHALLENGE_CHOICES), key="challenge_choice")
    configuration_keys = CHALLENGE_CHOICES[choice]

    images: dict[str, Any] = {}
    sample_errors: dict[str, str] = {}
    cards = st.columns(4)
    for column, domain in zip(cards, DOMAIN_ORDER, strict=True):
        sample = samples[selected_class][domain]
        with column:
            with st.spinner(f"Loading {DOMAIN_LABELS[domain]} sample..."):
                image, error = load_prepared_sample(sample)
            if error:
                sample_errors[domain] = error
                st.error(error, icon=":material/error:")
            else:
                images[domain] = image
                st.image(
                    image,
                    caption=f"{DOMAIN_LABELS[domain]} · {pretty_class(selected_class)}",
                    width="stretch",
                )
                st.caption("Known label verified from the public mirror.")

    if sample_errors:
        st.warning("Fix the sample download error above before running this challenge.")
        return
    challenge_fingerprint = f"{selected_class}|{'|'.join(configuration_keys)}"
    run = st.button(
        "Run selected models across 4 domains",
        type="primary",
        icon=":material/play_arrow:",
        key="challenge_run_button",
        width="stretch",
    )
    run_id = result_key(challenge_fingerprint, configuration_keys)
    if run:
        st.session_state["challenge_run"] = {
            "run_id": run_id,
            "configuration_keys": list(configuration_keys),
            "outcomes": run_configurations(specs, configuration_keys, images, class_names),
        }
    stored = st.session_state.get("challenge_run")
    if not stored or stored.get("run_id") != run_id:
        return

    st.subheader("Predictions by domain")
    rows: list[dict[str, str]] = []
    confidence_rows: list[dict[str, Any]] = []
    for key in configuration_keys:
        row: dict[str, str] = {"Configuration": specs[key].label}
        outcome = stored["outcomes"].get(key, {})
        for domain in DOMAIN_ORDER:
            if "error" in outcome:
                row[DOMAIN_LABELS[domain]] = f"Error: {outcome['error']}"
                continue
            top_one = outcome["predictions"][domain][0]
            correct = top_one["class"] == selected_class
            state = "Correct" if correct else "Incorrect"
            row[DOMAIN_LABELS[domain]] = (
                f"{pretty_class(str(top_one['class']))} · {float(top_one['probability']):.1%} · {state}"
            )
            confidence_rows.append(
                {
                    "Configuration": specs[key].label,
                    "Domain": DOMAIN_LABELS[domain],
                    "Top-1 confidence (%)": float(top_one["probability"]) * 100,
                    "Correct": correct,
                }
            )
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    if confidence_rows:
        st.subheader("Top-1 confidence by domain")
        confidence_frame = pd.DataFrame(confidence_rows)
        chart = (
            alt.Chart(confidence_frame)
            .mark_line(point=True)
            .encode(
                x=alt.X("Domain:N", sort=[DOMAIN_LABELS[item] for item in DOMAIN_ORDER], title="Image domain"),
                y=alt.Y("Top-1 confidence (%):Q", scale=alt.Scale(domain=[0, 100]), title="Top-1 confidence (%)"),
                color=alt.Color("Configuration:N", title="Configuration"),
                tooltip=[
                    alt.Tooltip("Configuration:N"),
                    alt.Tooltip("Domain:N"),
                    alt.Tooltip("Top-1 confidence (%):Q", format=".2f"),
                    alt.Tooltip("Correct:N"),
                ],
            )
            .properties(height=330)
        )
        st.altair_chart(chart)
        st.caption("This chart is confidence for four individual images, not test-set accuracy.")

    st.subheader("Connection to the full held-out evaluation")
    aggregate_rows: list[pd.DataFrame] = []
    for key in configuration_keys:
        frame = selected_configuration_rows(results, specs[key])
        frame["Configuration"] = specs[key].label
        aggregate_rows.append(frame[["Configuration", "Domain", "Accuracy (%)", "Macro-F1"]])
    st.dataframe(pd.concat(aggregate_rows, ignore_index=True), hide_index=True)
    st.caption("The table uses complete held-out partitions. Individual examples above illustrate behavior but do not replace those aggregate metrics.")


def render_model_comparison(
    specs: Mapping[str, CheckpointSpec],
    samples: Mapping[str, Mapping[str, SampleSpec]],
    class_names: Sequence[str],
) -> None:
    st.header("Model Comparison")
    st.caption("Run the same image through all four real checkpoints, then compare architecture and transfer-learning choices.")
    image, metadata, fingerprint = image_picker("comparison", samples)
    if image is None or fingerprint is None:
        return
    preview, action = st.columns((1, 1), gap="large")
    with preview:
        render_image_context(image, metadata, "Comparison image")
    with action:
        st.caption("All four predictions use the same deterministic archived test transform. Models are loaded and released sequentially on CPU.")
        run = st.button(
            "Compare all 4 models",
            type="primary",
            icon=":material/compare:",
            key="comparison_run_button",
            width="stretch",
        )
    run_id = result_key(fingerprint, CONFIGURATION_ORDER)
    if run:
        st.session_state["comparison_run"] = {
            "run_id": run_id,
            "outcomes": run_configurations(specs, CONFIGURATION_ORDER, {"selected": image}, class_names),
        }
    stored = st.session_state.get("comparison_run")
    if not stored or stored.get("run_id") != run_id:
        return

    outcomes = stored["outcomes"]
    st.subheader("Architecture comparison")
    architecture_columns = st.columns(2)
    with architecture_columns[0]:
        render_pair_comparison(
            "Linear Probe",
            "ResNet18",
            valid_top_one(outcomes, "resnet18_linear_probe", "selected"),
            "ResNet50",
            valid_top_one(outcomes, "resnet50_linear_probe", "selected"),
            top_five_for(outcomes, "resnet18_linear_probe", "selected"),
            top_five_for(outcomes, "resnet50_linear_probe", "selected"),
        )
    with architecture_columns[1]:
        render_pair_comparison(
            "Fine-tune Last Block",
            "ResNet18",
            valid_top_one(outcomes, "resnet18_finetune_last_block", "selected"),
            "ResNet50",
            valid_top_one(outcomes, "resnet50_finetune_last_block", "selected"),
            top_five_for(outcomes, "resnet18_finetune_last_block", "selected"),
            top_five_for(outcomes, "resnet50_finetune_last_block", "selected"),
        )

    st.subheader("Training strategy comparison")
    strategy_columns = st.columns(2)
    with strategy_columns[0]:
        render_pair_comparison(
            "ResNet18",
            "Linear Probe",
            valid_top_one(outcomes, "resnet18_linear_probe", "selected"),
            "Fine-tune",
            valid_top_one(outcomes, "resnet18_finetune_last_block", "selected"),
            top_five_for(outcomes, "resnet18_linear_probe", "selected"),
            top_five_for(outcomes, "resnet18_finetune_last_block", "selected"),
        )
    with strategy_columns[1]:
        render_pair_comparison(
            "ResNet50",
            "Linear Probe",
            valid_top_one(outcomes, "resnet50_linear_probe", "selected"),
            "Fine-tune",
            valid_top_one(outcomes, "resnet50_finetune_last_block", "selected"),
            top_five_for(outcomes, "resnet50_linear_probe", "selected"),
            top_five_for(outcomes, "resnet50_finetune_last_block", "selected"),
        )
    st.caption("A confidence difference is shown only for this image; higher confidence alone does not prove that one configuration is better overall.")


def render_full_experiment(
    results: pd.DataFrame,
    specs: Mapping[str, CheckpointSpec],
) -> None:
    st.header("Full Experiment")
    st.caption("Individual examples above illustrate model behavior. The figures here summarize complete held-out test partitions.")
    selected_key = st.selectbox(
        "Focus configuration",
        list(CONFIGURATION_ORDER),
        format_func=lambda key: specs[key].label,
        key="full_experiment_config",
    )
    selected_spec = specs[selected_key]
    selected = selected_configuration_rows(results, selected_spec)

    table_column, chart_column = st.columns(2, gap="large")
    with table_column:
        st.subheader("Accuracy")
        accuracy_table = configuration_table(results, specs, "accuracy") * 100
        st.dataframe(
            accuracy_table,
            column_config={
                domain: st.column_config.NumberColumn(domain, format="%.2f%%")
                for domain in accuracy_table.columns
            },
        )
        st.subheader("Macro-F1")
        macro_table = configuration_table(results, specs, "macro_f1")
        st.dataframe(
            macro_table,
            column_config={
                domain: st.column_config.NumberColumn(domain, format="%.4f")
                for domain in macro_table.columns
            },
        )
    with chart_column:
        st.subheader(f"Accuracy by domain — {selected_spec.label}")
        accuracy_chart = (
            alt.Chart(selected)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("Domain:N", sort=[DOMAIN_LABELS[item] for item in DOMAIN_ORDER], title="Test domain"),
                y=alt.Y("Accuracy (%):Q", scale=alt.Scale(domain=[0, 100]), title="Accuracy (%)"),
                color=alt.Color("Domain:N", legend=None),
                tooltip=[
                    alt.Tooltip("Domain:N"),
                    alt.Tooltip("Accuracy (%):Q", format=".2f"),
                    alt.Tooltip("Macro-F1:Q", format=".4f"),
                ],
            )
            .properties(height=330)
        )
        st.altair_chart(accuracy_chart)

    drop_rows = selected[selected["test_domain"] != "Real_World"].copy()
    st.subheader("Source-to-target accuracy drop")
    drop_chart = (
        alt.Chart(drop_rows)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color="#D97706")
        .encode(
            x=alt.X("Domain:N", sort=["Product", "Art", "Clipart"], title="Target domain"),
            y=alt.Y("Drop from Real-World (pp):Q", title="Accuracy drop (percentage points)"),
            tooltip=[
                alt.Tooltip("Domain:N"),
                alt.Tooltip("Drop from Real-World (pp):Q", format=".2f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(drop_chart)

    st.subheader("All four configurations")
    comparison = results.copy()
    comparison["Configuration"] = comparison.apply(
        lambda row: specs[f"{row['model']}_{row['mode']}"].label,
        axis=1,
    )
    comparison["Domain"] = comparison["test_domain"].map(DOMAIN_LABELS)
    comparison["Accuracy (%)"] = comparison["accuracy"] * 100
    ordered_labels = [specs[key].label for key in CONFIGURATION_ORDER]
    all_chart = (
        alt.Chart(comparison)
        .mark_bar()
        .encode(
            x=alt.X("Domain:N", sort=[DOMAIN_LABELS[item] for item in DOMAIN_ORDER], title="Test domain"),
            y=alt.Y("Accuracy (%):Q", scale=alt.Scale(domain=[0, 100]), title="Accuracy (%)"),
            color=alt.Color("Configuration:N", sort=ordered_labels, title="Configuration"),
            xOffset="Configuration:N",
            tooltip=[
                alt.Tooltip("Configuration:N"),
                alt.Tooltip("Domain:N"),
                alt.Tooltip("Accuracy (%):Q", format=".2f"),
                alt.Tooltip("macro_f1:Q", title="Macro-F1", format=".4f"),
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(all_chart)
    st.caption("All reported measurements use one training seed and fixed partitions. They do not establish statistical significance.")


def render_app() -> None:
    initialise_state()
    try:
        results = load_results()
        class_names = load_class_names()
        specs = cached_checkpoint_specs()
        samples = cached_sample_specs()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        st.error(f"The interactive demo could not load its archived metadata: {error}", icon=":material/error:")
        st.stop()

    with st.sidebar:
        st.title("Experiment")
        page = st.radio(
            "Explore",
            ["Playground", "Domain Shift Challenge", "Model Comparison", "Full Experiment"],
            key="page",
        )
        st.caption("65 classes · source training domain: Real-World · CPU inference")
        st.caption("Checkpoints are SHA-256 verified before loading and are not committed to Git.")

    render_header()
    if page == "Playground":
        render_playground(specs, samples, class_names)
    elif page == "Domain Shift Challenge":
        render_domain_shift_challenge(specs, samples, class_names, results)
    elif page == "Model Comparison":
        render_model_comparison(specs, samples, class_names)
    else:
        render_full_experiment(results, specs)


render_app()
