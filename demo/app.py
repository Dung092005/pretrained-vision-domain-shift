"""Interactive dashboard for the archived Office-Home domain-shift experiment."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

try:
    from demo.inference import (
        CheckpointError,
        compatible_candidates,
        discover_checkpoints,
        load_model,
        predict_image,
    )
except ModuleNotFoundError:  # Supports `streamlit run demo/app.py` as well.
    from inference import (
        CheckpointError,
        compatible_candidates,
        discover_checkpoints,
        load_model,
        predict_image,
    )


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "results_all.csv"
CLASS_MAPPING_PATH = ROOT / "results" / "class_to_idx.json"
DOMAIN_EXAMPLES_PATH = ROOT / "figures" / "domain_examples.png"

DOMAIN_ORDER = ["Real_World", "Product", "Art", "Clipart"]
DOMAIN_LABELS = {
    "Real_World": "Real-World",
    "Product": "Product",
    "Art": "Art",
    "Clipart": "Clipart",
}
MODEL_LABELS = {"ResNet18": "resnet18", "ResNet50": "resnet50"}
MODE_LABELS = {
    "Linear Probe": "linear_probe",
    "Fine-tune Last Block": "finetune_last_block",
}
CONFIGURATION_ORDER = [
    ("resnet18", "linear_probe"),
    ("resnet18", "finetune_last_block"),
    ("resnet50", "linear_probe"),
    ("resnet50", "finetune_last_block"),
]


st.set_page_config(
    page_title="Vision domain-shift demo",
    page_icon=":material/visibility:",
    layout="wide",
)


def model_label(model_name: str) -> str:
    return "ResNet18" if model_name == "resnet18" else "ResNet50"


def mode_label(mode: str) -> str:
    return "Linear Probe" if mode == "linear_probe" else "Fine-tune Last Block"


def configuration_label(model_name: str, mode: str) -> str:
    return f"{model_label(model_name)} — {mode_label(mode)}"


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    """Load the compact, saved evaluation table rather than recomputing metrics."""
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Missing archived results: {RESULTS_PATH}")
    results = pd.read_csv(RESULTS_PATH)
    required_columns = {
        "model",
        "mode",
        "test_domain",
        "accuracy",
        "macro_f1",
        "accuracy_drop_pp",
        "experiment",
    }
    missing = required_columns.difference(results.columns)
    if missing:
        raise ValueError(f"The archived results are missing columns: {', '.join(sorted(missing))}")
    return results


@st.cache_data(show_spinner=False)
def load_class_names() -> list[str]:
    """Return class names in the exact saved classifier-index order."""
    mapping = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]


@st.cache_data(show_spinner=False)
def load_checkpoint_candidates() -> list[object]:
    return discover_checkpoints(ROOT)


@st.cache_resource(show_spinner="Loading selected checkpoint on CPU...")
def cached_model(checkpoint_path: str, selected_model: str, class_names: tuple[str, ...]):
    return load_model(checkpoint_path, selected_model, list(class_names))


def selected_configuration_rows(results: pd.DataFrame, model_name: str, mode: str) -> pd.DataFrame:
    rows = results[(results["model"] == model_name) & (results["mode"] == mode)].copy()
    rows = rows.set_index("test_domain").reindex(DOMAIN_ORDER).reset_index()
    if rows["accuracy"].isna().any():
        raise ValueError("The selected configuration does not have all four domain evaluations.")
    rows["Domain"] = rows["test_domain"].map(DOMAIN_LABELS)
    rows["Accuracy (%)"] = rows["accuracy"] * 100
    rows["Macro-F1"] = rows["macro_f1"]
    rows["Drop from Real-World (pp)"] = rows["accuracy_drop_pp"]
    return rows


def configuration_table(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = results.copy()
    data["Configuration"] = data.apply(
        lambda row: configuration_label(row["model"], row["mode"]), axis=1
    )
    data["Domain"] = data["test_domain"].map(DOMAIN_LABELS)
    ordered_configurations = [configuration_label(model, mode) for model, mode in CONFIGURATION_ORDER]
    table = data.pivot(index="Configuration", columns="Domain", values=metric)
    return table.reindex(index=ordered_configurations, columns=[DOMAIN_LABELS[item] for item in DOMAIN_ORDER])


def render_header() -> None:
    st.title("Pretrained Vision Models Across Image Domains")
    st.caption("Exploring how ImageNet-pretrained ResNet models behave under visual domain shift.")
    st.write(
        "The models are trained on Real-World images and evaluated on Real-World, "
        "Product, Art, and Clipart domains."
    )


def render_overview(results: pd.DataFrame) -> None:
    st.header("Overview")
    st.write(
        "A model trained on real photographs may perform differently when the same "
        "objects appear as product photos, artwork, or clipart. This demo compares that change."
    )

    with st.container(horizontal=True):
        st.metric("Dataset", "Office-Home", border=True)
        st.metric("Classes", "65", border=True)
        st.metric("Images after deduplication", "15,176", border=True)
        st.metric("Domains", "4", border=True)
        st.metric("Configurations", str(results["experiment"].nunique()), border=True)

    st.subheader("Experiment pipeline")
    source, models, strategies, evaluation = st.columns(4, border=True)
    with source:
        st.markdown(":material/photo_camera: **Real-World training data**")
        st.caption("Only the source-domain training subset updates model parameters.")
    with models:
        st.markdown(":material/memory: **ResNet18 / ResNet50**")
        st.caption("Both begin with ImageNet-pretrained weights and a 65-class head.")
    with strategies:
        st.markdown(":material/tune: **Transfer-learning strategy**")
        st.caption("Linear probe, or fine-tune layer4 plus the classification head.")
    with evaluation:
        st.markdown(":material/analytics: **Held-out evaluation**")
        st.caption("Real-World, Product, Art, and Clipart test partitions.")

    st.caption(
        "Model selection uses Real-World validation macro-F1. Target-domain training and validation labels are not used."
    )


def render_results(results: pd.DataFrame, model_name: str, mode: str) -> None:
    st.header("Experiment results")
    selected = selected_configuration_rows(results, model_name, mode)
    config_name = configuration_label(model_name, mode)
    st.caption(f"Selected configuration: {config_name}")

    table_col, chart_col = st.columns((1, 1), gap="large")
    with table_col:
        st.subheader("Accuracy table")
        accuracy_table = configuration_table(results, "accuracy") * 100
        st.dataframe(
            accuracy_table,
            column_config={
                domain: st.column_config.NumberColumn(domain, format="%.2f%%")
                for domain in accuracy_table.columns
            },
        )
        st.subheader("Macro-F1 table")
        macro_table = configuration_table(results, "macro_f1")
        st.dataframe(
            macro_table,
            column_config={
                domain: st.column_config.NumberColumn(domain, format="%.4f")
                for domain in macro_table.columns
            },
        )

    with chart_col:
        st.subheader("Accuracy by domain")
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

    st.subheader("Source-to-target accuracy drop")
    target_rows = selected[selected["test_domain"] != "Real_World"].copy()
    drop_chart = (
        alt.Chart(target_rows)
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

    st.subheader("All configurations")
    comparison = results.copy()
    comparison["Configuration"] = comparison.apply(
        lambda row: configuration_label(row["model"], row["mode"]), axis=1
    )
    comparison["Domain"] = comparison["test_domain"].map(DOMAIN_LABELS)
    comparison["Accuracy (%)"] = comparison["accuracy"] * 100
    all_configurations = [configuration_label(model, mode) for model, mode in CONFIGURATION_ORDER]
    comparison_chart = (
        alt.Chart(comparison)
        .mark_bar()
        .encode(
            x=alt.X("Domain:N", sort=[DOMAIN_LABELS[item] for item in DOMAIN_ORDER], title="Test domain"),
            y=alt.Y("Accuracy (%):Q", scale=alt.Scale(domain=[0, 100]), title="Accuracy (%)"),
            color=alt.Color("Configuration:N", sort=all_configurations, title="Configuration"),
            xOffset="Configuration:N",
            tooltip=[
                alt.Tooltip("Configuration:N"),
                alt.Tooltip("Domain:N"),
                alt.Tooltip("Accuracy (%):Q", format=".2f"),
                alt.Tooltip("macro_f1:Q", title="Macro-F1", format=".4f"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(comparison_chart)

    st.subheader("What we observed")
    st.markdown(
        "- Performance decreases when evaluation moves away from the Real-World source domain.\n"
        "- Clipart is the most difficult domain in this experiment.\n"
        "- ResNet50 generally performs better than ResNet18.\n"
        "- Fine-tuning does not always improve cross-domain performance."
    )
    st.caption(
        "These are single-seed results on fixed partitions. They do not establish statistical significance or universal behavior."
    )


def render_domain_comparison(results: pd.DataFrame, model_name: str, mode: str) -> None:
    st.header("Domain comparison")
    selected = selected_configuration_rows(results, model_name, mode)
    st.caption(f"Selected configuration: {configuration_label(model_name, mode)}")

    if DOMAIN_EXAMPLES_PATH.is_file():
        st.image(
            DOMAIN_EXAMPLES_PATH,
            caption="Saved Office-Home examples for several classes across the four visual domains.",
            width="stretch",
        )

    st.subheader("Accuracy and change from Real-World")
    cards = st.columns(4)
    for card, (_, row) in zip(cards, selected.iterrows(), strict=True):
        with card:
            delta = None if row["test_domain"] == "Real_World" else f"-{row['accuracy_drop_pp']:.2f} pp"
            st.metric(
                str(row["Domain"]),
                f"{row['Accuracy (%)']:.2f}%",
                delta=delta,
                delta_color="normal",
                border=True,
            )

    comparison_chart = (
        alt.Chart(selected)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Domain:N", sort=[DOMAIN_LABELS[item] for item in DOMAIN_ORDER], title="Domain"),
            y=alt.Y("Accuracy (%):Q", scale=alt.Scale(domain=[0, 100]), title="Accuracy (%)"),
            color=alt.Color("Domain:N", legend=None),
            tooltip=[
                alt.Tooltip("Domain:N"),
                alt.Tooltip("Accuracy (%):Q", format=".2f"),
                alt.Tooltip("Drop from Real-World (pp):Q", format=".2f"),
            ],
        )
        .properties(height=330)
    )
    st.altair_chart(comparison_chart)
    st.caption("Drops are measured against the held-out Real-World test accuracy, in percentage points.")


def render_live_prediction(model_name: str, mode: str) -> None:
    st.header("Live prediction")
    candidates = compatible_candidates(load_checkpoint_candidates(), model_name, mode)

    if not candidates:
        st.warning(
            "Live inference requires the trained checkpoint files. The experiment results below are loaded from the completed run.",
            icon=":material/info:",
        )
        st.caption(
            "No compatible local best.pt, last.pt, .pt, .pth, or .ckpt file was found. "
            "The Google Drive location referenced by the project is not downloaded automatically."
        )
        return

    st.caption("A compatible local checkpoint was found. Inference runs on CPU and loads only the selected model.")
    selected_candidate = st.selectbox(
        "Checkpoint",
        candidates,
        format_func=lambda candidate: candidate.label,
        key="checkpoint_choice",
    )
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload a JPG, JPEG, or PNG image to run the selected checkpoint.", icon=":material/upload:")
        return

    from PIL import Image

    image = Image.open(uploaded).convert("RGB")
    image_col, prediction_col = st.columns((1, 1), gap="large")
    with image_col:
        st.image(image, caption="Uploaded image", width="stretch")

    with prediction_col:
        with st.form("prediction_form"):
            submitted = st.form_submit_button("Predict image", type="primary", icon=":material/play_arrow:")
        if not submitted:
            st.caption("The archived evaluation preprocessing uses RGB conversion, resize-to-256, center crop 224, and ImageNet normalization.")
            return
        try:
            class_names = load_class_names()
            model, torch = cached_model(str(selected_candidate.path), model_name, tuple(class_names))
            predictions = predict_image(model, torch, image, class_names)
        except (CheckpointError, RuntimeError, OSError) as error:
            st.error(f"Unable to run this checkpoint: {error}", icon=":material/error:")
            return

        top_prediction = predictions[0]
        st.metric("Top prediction", str(top_prediction["class"]), f"{float(top_prediction['probability']) * 100:.2f}% confidence", border=True)
        top_five = pd.DataFrame(predictions)
        top_five["probability"] = top_five["probability"] * 100
        st.dataframe(
            top_five.rename(columns={"class": "Class", "probability": "Probability (%)"}),
            column_config={"Probability (%)": st.column_config.NumberColumn(format="%.2f%%")},
            hide_index=True,
        )


def render_app() -> None:
    try:
        results = load_results()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        st.error(f"The archived result files could not be loaded: {error}", icon=":material/error:")
        st.stop()

    with st.sidebar:
        st.title("Experiment controls")
        page = st.radio(
            "Sections",
            ["Overview", "Live Prediction", "Domain Comparison", "Experiment Results"],
            key="page",
            width="stretch",
        )
        st.selectbox("Model", list(MODEL_LABELS), key="model_choice")
        st.selectbox("Training strategy", list(MODE_LABELS), key="mode_choice")
        st.caption("Saved single-seed results · source domain: Real-World")

    render_header()
    selected_model = MODEL_LABELS[st.session_state["model_choice"]]
    selected_mode = MODE_LABELS[st.session_state["mode_choice"]]

    if page == "Overview":
        render_overview(results)
    elif page == "Live Prediction":
        render_live_prediction(selected_model, selected_mode)
    elif page == "Domain Comparison":
        render_domain_comparison(results, selected_model, selected_mode)
    else:
        render_results(results, selected_model, selected_mode)


render_app()
