"""CBS4702 Final Project - Emotion Classification demo app.

Run locally:
    streamlit run app/app.py

The app loads the model artefacts produced by src/02_train_models.py and
src/03_train_transformer.py, so those scripts must be run once before the
first launch.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Emotion Classification Demo", page_icon="🎭",
                   layout="wide")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL_EMOJI = {"sadness": "😢", "joy": "😄", "love": "❤️",
               "anger": "😠", "fear": "😨", "surprise": "😲"}
MODEL_LABELS = {
    "B1_BoW_LR": "B1 · Bag-of-Words + Logistic Regression (baseline)",
    "B2_TFIDF_NB": "B2 · TF-IDF + Multinomial Naive Bayes (baseline)",
    "B3_TFIDF_LR": "B3 · TF-IDF(1-gram) + Logistic Regression (baseline)",
    "I1_word12_cw_LR": "I1 · TF-IDF(1-2gram) + class-weighted LR (improved)",
    "I2_union_cw_LR": "I2 · word+char n-gram union + class-weighted LR (improved)",
    "I3_union_cw_SVC": "I3 · word+char n-gram union + Linear SVC (improved)",
    "I4_union_LSA_LR": "I4 · LSA(300) + LR (improved)",
    "I5_DistilBERT": "I5 · fine-tuned DistilBERT (improved)",
}
DEFAULT_MODELS = ["B1_BoW_LR", "I3_union_cw_SVC", "I5_DistilBERT"]


# --------------------------------------------------------------------------- #
# Artefacts
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_sklearn_model(name: str):
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_transformer():
    path = os.path.join(MODEL_DIR, "distilbert_emotion")
    if not os.path.exists(os.path.join(path, "config.json")):
        return None, None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception:
        return None, None
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    return (model, tok) if model is not None else (None, None)


@st.cache_data(show_spinner=False)
def load_csv(name: str, folder: str = RESULT_DIR):
    path = os.path.join(folder, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict_proba(name: str, texts: list[str]) -> np.ndarray | None:
    """Return a (n_texts, 6) probability matrix for any supported model."""
    if name == "I5_DistilBERT":
        model, tok = load_transformer()
        if model is None:
            return None
        import torch
        device = next(model.parameters()).device
        enc = tok(texts, truncation=True, max_length=64, padding="max_length",
                  return_tensors="pt")
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device)).logits
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()

    pipe = load_sklearn_model(name)
    if pipe is None:
        return None
    clf = pipe.named_steps["clf"]
    X = pipe.named_steps["vec"].transform(texts)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    return softmax(clf.decision_function(X))


def clean_input(raw: str) -> tuple[str | None, str | None]:
    """Validate user input. Returns (cleaned_text, error_message)."""
    if raw is None or not raw.strip():
        return None, "Please type an English sentence first (the box is empty)."
    text = " ".join(raw.split())
    if not any(ch.isalpha() for ch in text):
        return None, "Your input has no letters. Please enter an English sentence."
    if len(text.split()) < 2:
        return None, "Please enter at least two words so the model has context."
    return text[:500], None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("Settings")
available = [m for m in MODEL_LABELS
             if os.path.exists(os.path.join(MODEL_DIR, f"{m}.joblib"))
             or (m == "I5_DistilBERT"
                 and os.path.exists(os.path.join(MODEL_DIR, "distilbert_emotion",
                                                 "config.json")))]
if not available:
    st.error("No model artefacts found. Run `python3 src/02_train_models.py` and "
             "`python3 src/03_train_transformer.py` first.")
    st.stop()

selected_models = st.sidebar.multiselect(
    "Models to compare", options=available,
    default=[m for m in DEFAULT_MODELS if m in available],
    format_func=lambda m: MODEL_LABELS[m])
if not selected_models:
    st.sidebar.warning("Select at least one model.")

st.sidebar.markdown("---")
st.sidebar.caption("Models are trained on the 16,000-example training split of "
                   "`dair-ai/emotion`; the test split is used only for evaluation.")

# --------------------------------------------------------------------------- #
# Header + prediction
# --------------------------------------------------------------------------- #
st.title("Emotion Classification of English Short Text")
st.caption("CBS4702 Advanced Topics in Quantitative Language Studies · Final Project")

tab_predict, tab_compare, tab_cm, tab_explore = st.tabs(
    ["Predict", "Model comparison", "Confusion matrix", "Example explorer"])

with tab_predict:
    examples = [
        "i feel like i am drowning and nobody notices",
        "today was perfect, i got the internship and celebrated with my friends",
        "i am so angry that they cancelled the plan at the last minute",
        "i can't stop worrying about the results",
        "i did not expect that at all, i am totally stunned",
        "my grandmother hugged me and told me she is proud of me",
        "i am not sad, i am just tired",
    ]
    col_input, col_side = st.columns([2, 1])
    with col_input:
        default_text = st.selectbox("Try an example, or write your own below:",
                                    ["(write my own)"] + examples)
        user_text = st.text_area(
            "Your English sentence",
            value="" if default_text == "(write my own)" else default_text,
            height=110, placeholder="e.g. i feel so lonely tonight")
        go = st.button("Classify", type="primary")

    with col_side:
        st.markdown("**Notes**")
        st.markdown("- Empty or non-English input is rejected with a hint.\n"
                    "- Input is trimmed to 500 characters.\n"
                    "- Probabilities come straight from each model; a low top "
                    "probability means the model is uncertain.")

    if go:
        text, err = clean_input(user_text)
        if err:
            st.warning(err)
        elif not selected_models:
            st.warning("Please select at least one model in the sidebar.")
        else:
            st.markdown(f"**Input:** {text}")
            cols = st.columns(len(selected_models))
            for col, name in zip(cols, selected_models):
                probs = predict_proba(name, [text])
                with col:
                    if probs is None:
                        st.error(f"{name}: artefact not found.")
                        continue
                    p = probs[0]
                    k = int(np.argmax(p))
                    st.metric(label=LABEL_EMOJI[LABEL_NAMES[k]] + " " +
                              LABEL_NAMES[k].upper(), value=f"{p[k]:.1%}",
                              help=MODEL_LABELS[name])
                    st.progress(float(p[k]))
                    st.bar_chart(pd.DataFrame({"probability": p},
                                              index=[f"{e} {c}" for e, c in
                                                     zip([LABEL_EMOJI[c] for c in LABEL_NAMES],
                                                         LABEL_NAMES)]))
                    st.caption(MODEL_LABELS[name])

            if len(selected_models) > 1:
                agree = {n: LABEL_NAMES[int(np.argmax(predict_proba(n, [text])[0]))]
                         for n in selected_models}
                if len(set(agree.values())) == 1:
                    st.success(f"All selected models agree on **{list(agree.values())[0]}**.")
                else:
                    st.info("Models disagree: " +
                            " · ".join(f"{n} → {v}" for n, v in agree.items()))

# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
with tab_compare:
    comp = load_csv("model_comparison.csv")
    per_class = load_csv("per_class_metrics.csv")
    if comp is None or per_class is None:
        st.info("Run `python3 src/02_train_models.py` to generate the comparison tables.")
    else:
        st.subheader("Overall accuracy and macro-F1 on the test split (n = 2,000)")
        show = comp[["model", "family", "accuracy", "macro_f1"]].copy()
        show.columns = ["model", "family", "accuracy", "macro-F1"]
        st.dataframe(show.style.format({"accuracy": "{:.3f}", "macro-F1": "{:.3f}"})
                     .background_gradient(subset=["macro-F1"], cmap="Blues"),
                     use_container_width=True, hide_index=True)

        st.subheader("Per-class F1")
        pivot = per_class.pivot(index="model", columns="class", values="f1")
        pivot = pivot[[c for c in LABEL_NAMES if c in pivot.columns]]
        st.bar_chart(pivot)
        st.caption("Surprise and love are consistently the hardest classes: they are "
                   "the smallest classes and overlap with fear / joy in wording.")

# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #
with tab_cm:
    cms = sorted([f for f in os.listdir(RESULT_DIR) if f.startswith("cm_")]) \
        if os.path.isdir(RESULT_DIR) else []
    if not cms:
        st.info("No confusion matrices found. Run the training scripts first.")
    else:
        pick = st.selectbox("Model", [c[3:-4] for c in cms], index=len(cms) - 1)
        cm = pd.read_csv(os.path.join(RESULT_DIR, f"cm_{pick}.csv"), index_col=0)
        st.markdown(f"**Confusion matrix — {pick}** (rows = gold, columns = predicted)")
        st.dataframe(cm.style.background_gradient(cmap="Blues", axis=None),
                     use_container_width=True)
        row_norm = cm.div(cm.sum(axis=1), axis=0)
        st.markdown("**Row-normalised (recall per class)**")
        st.dataframe((row_norm * 100).round(1).astype(str) + " %",
                     use_container_width=True)

# --------------------------------------------------------------------------- #
# Example explorer (incl. errors)
# --------------------------------------------------------------------------- #
with tab_explore:
    files = sorted([f for f in os.listdir(RESULT_DIR)
                    if f.startswith("preds_")]) if os.path.isdir(RESULT_DIR) else []
    if not files:
        st.info("No test-set predictions found. Run the training scripts first.")
    else:
        pick = st.selectbox("Model", [f[6:-4] for f in files])
        preds = pd.read_csv(os.path.join(RESULT_DIR, f"preds_{pick}.csv"))
        preds["gold_name"] = preds["gold"].map(dict(enumerate(LABEL_NAMES)))
        preds["pred_name"] = preds["pred"].map(dict(enumerate(LABEL_NAMES)))
        preds["correct"] = preds["gold"] == preds["pred"]

        mode = st.radio("Show", ["Correct predictions", "Errors only", "All"],
                        horizontal=True)
        if mode == "Errors only":
            view = preds[~preds["correct"]]
        elif mode == "Correct predictions":
            view = preds[preds["correct"]]
        else:
            view = preds
        st.write(f"{len(view)} examples (accuracy on this view: "
                 f"{view['correct'].mean():.1%})")
        st.dataframe(view[["text", "gold_name", "pred_name"]].rename(
            columns={"text": "text", "gold_name": "true label",
                     "pred_name": "predicted"}),
            use_container_width=True, height=430, hide_index=True)
        st.download_button("Download these predictions (CSV)",
                           view.to_csv(index=False).encode("utf-8"),
                           file_name=f"predictions_{pick}.csv")
