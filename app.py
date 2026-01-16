from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.app_state import (
    ARTIFACT_PATH,
    SessionKeys,
    clear_uploaded_dataset,
    compute_cleaned,
    get_dataset_label,
    get_feature_columns,
    get_raw_df,
    require_target_column,
    set_cleaning_cfg,
    set_uploaded_dataset,
)
from src.cleaning import CleaningConfig
from src.constants import TARGET_COL
from src.modeling import TrainConfig, load_artifact, save_artifact, train_and_evaluate

st.set_page_config(page_title="Credit Risk", layout="wide", page_icon="📊")


def _kpi(label: str, value: object, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def _target_bar(df: pd.DataFrame) -> None:
    vc = (
        df[TARGET_COL]
        .value_counts(dropna=False)
        .rename_axis(TARGET_COL)
        .reset_index(name="count")
        .assign(pct=lambda x: (x["count"] / x["count"].sum() * 100).round(2))
    )
    chart = (
        alt.Chart(vc)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X(f"{TARGET_COL}:N", sort="-y", title=None),
            y=alt.Y("count:Q", title="Count"),
            tooltip=[TARGET_COL, "count", "pct"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)


def _hist(df: pd.DataFrame, col: str) -> None:
    s = pd.to_numeric(df[col], errors="coerce")
    tmp = pd.DataFrame({col: s}).dropna()
    if tmp.empty:
        st.info("No numeric values to plot.")
        return
    chart = (
        alt.Chart(tmp)
        .mark_bar(opacity=0.9, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30), title=col),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="count")],
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


def _cat_counts(df: pd.DataFrame, col: str, top_k: int = 12) -> None:
    vc = df[col].astype(str).fillna("NA").value_counts().head(top_k).rename_axis(col).reset_index(name="count")
    chart = (
        alt.Chart(vc)
        .mark_bar()
        .encode(
            y=alt.Y(f"{col}:N", sort="-x", title=None),
            x=alt.X("count:Q", title="Count"),
            tooltip=[col, "count"],
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


st.title("Credit Risk")
st.caption("End‑to‑end workflow: dataset → exploration → preprocessing → training → prediction.")

# Sidebar: dataset + navigation
with st.sidebar:
    st.subheader("Workspace")
    uploaded = st.file_uploader("Dataset (.xlsx/.xls/.csv)", type=["xlsx", "xls", "csv"])
    if uploaded is not None:
        set_uploaded_dataset(file_bytes=uploaded.getvalue(), filename=uploaded.name)

    st.caption("Active dataset")
    st.code(get_dataset_label())
    if st.button("Reset dataset", type="secondary", use_container_width=True):
        clear_uploaded_dataset()
        st.rerun()

    st.divider()
    section = st.radio(
        "Navigate",
        options=["Dashboard", "Explore", "Prepare", "Train", "Predict"],
        index=0,
        label_visibility="visible",
    )

# Load data
df = get_raw_df()
require_target_column(df)


if section == "Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    _kpi("Rows", int(df.shape[0]))
    _kpi("Columns", int(df.shape[1]))
    _kpi("Null cells", int(df.isna().sum().sum()))
    _kpi("Duplicates", int(df.duplicated().sum()))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Preview")
        st.dataframe(df.head(30), use_container_width=True)
    with right:
        st.subheader("Target distribution")
        _target_bar(df)

    st.divider()
    st.subheader("Data profile")
    with st.expander("Dtypes", expanded=False):
        st.json(df.dtypes.astype(str).to_dict())
    with st.expander("Missing values by column", expanded=False):
        missing_tbl = df.isna().sum().rename("missing").rename_axis("column").reset_index()
        st.dataframe(missing_tbl, use_container_width=True, hide_index=True)


elif section == "Explore":
    st.subheader("EDA")
    st.caption("Correlation matrix")

    num_cols, cat_cols = get_feature_columns(df)

    corr = df[num_cols].corr() if num_cols else pd.DataFrame()

    if corr.empty:
        st.info("No numeric columns available for correlation.")
    else:
        # NOTE: avoid pandas Styler background_gradient because it requires matplotlib.
        st.dataframe(corr.round(3), use_container_width=True)

    st.divider()
    st.subheader("Summary tables")
    a, b = st.columns(2)
    with a:
        st.caption("Numeric describe")
        st.dataframe(df[num_cols].describe().T if num_cols else pd.DataFrame(), use_container_width=True)
    with b:
        st.caption("Categorical describe")
        st.dataframe(df[cat_cols].describe().T if cat_cols else pd.DataFrame(), use_container_width=True)


elif section == "Prepare":
    st.subheader("Preprocessing")
    st.caption("Configure cleaning rules and generate a cleaned dataset for modeling.")

    num_cols, _cat_cols = get_feature_columns(df)

    with st.form("prep_form", border=True):
        a, b, c = st.columns([1.2, 1.2, 1])

        with a:
            st.markdown("**Missing values**")
            numeric_missing = st.radio("Numeric strategy", options=["median", "mean", "drop"], index=0, horizontal=True)
            categorical_missing = st.radio("Categorical strategy", options=["mode", "drop"], index=0, horizontal=True)

        with b:
            st.markdown("**Duplicates & outliers**")
            drop_dups = st.toggle("Drop duplicates", value=True)
            outlier_method = st.selectbox("Outlier filter", options=["none", "zscore", "mean_std"], index=0)
            outlier_cols = st.multiselect("Outlier columns", options=num_cols, default=num_cols[:1] if num_cols else [])

        with c:
            st.markdown("**Outlier thresholds**")
            z_thr = st.slider("Z-score", 1.0, 5.0, 3.0, 0.1)
            k_std = st.slider("k·std", 1.0, 5.0, 3.0, 0.1)

        run = st.form_submit_button("Run preprocessing", type="primary")

    if run:
        cfg = CleaningConfig(
            target_col=TARGET_COL,
            numeric_missing=str(numeric_missing),
            categorical_missing=str(categorical_missing),
            drop_duplicates=bool(drop_dups),
            outlier_method=str(outlier_method),
            outlier_cols=list(outlier_cols),
            zscore_threshold=float(z_thr),
            mean_std_k=float(k_std),
        )
        set_cleaning_cfg(cfg)
        st.success("Preprocessing configuration saved.")

    clean_df, report = compute_cleaned(df)

    m1, m2, m3, m4 = st.columns(4)
    _kpi("Rows (before)", report["rows_before"])
    _kpi("Rows (after)", report["rows_after"])
    _kpi("Nulls (after)", report["nulls_after"])
    _kpi("Removed rows", int(report["rows_before"]) - int(report["rows_after"]))

    st.divider()
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Cleaned preview")
        st.dataframe(clean_df.head(40), use_container_width=True)
    with right:
        st.subheader("Download")
        st.download_button(
            "Download cleaned CSV",
            data=clean_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
        with st.expander("Full preprocessing report", expanded=False):
            st.json(report)


elif section == "Train":
    st.subheader("Training lab")
    st.caption("Train multiple models on the cleaned dataset and optionally save the best to artifacts.")

    clean_df, _ = compute_cleaned(df)
    num_cols, cat_cols = get_feature_columns(clean_df)

    left, right = st.columns([1.2, 1])
    with left:
        test_size = st.slider("Holdout test size", 0.10, 0.50, 0.30, 0.05)
        seed = st.number_input("Random seed", value=42, step=1)
    with right:
        st.markdown("**Algorithms**")
        run_all = st.checkbox("Train all (recommended)", value=True)
        chosen = st.multiselect(
            "Select models",
            options=["HistGB", "RandomForest", "GradientBoosting"],
            default=["HistGB", "RandomForest", "GradientBoosting"],
            disabled=run_all,
        )

    models = ["HistGB", "RandomForest", "GradientBoosting"] if run_all else list(chosen)
    if not models:
        st.info("Select at least one model.")
        st.stop()

    if st.button("Start training", type="primary"):
        results: list[dict[str, object]] = []
        pipes: dict[str, dict[str, object]] = {}
        with st.spinner("Training..."):
            for name in models:
                pipe, metrics = train_and_evaluate(
                    clean_df,
                    numeric_cols=num_cols,
                    categorical_cols=cat_cols,
                    cfg=TrainConfig(
                        target_col=TARGET_COL,
                        test_size=float(test_size),
                        random_state=int(seed),
                        model_name=name,
                    ),
                )
                results.append(
                    {k: v for k, v in metrics.items() if k in {"model", "accuracy", "precision", "recall", "f1", "roc_auc"}}
                )
                pipes[name] = {"pipeline": pipe, "metrics": metrics}

        st.session_state[SessionKeys.MODEL_RESULTS] = results
        st.session_state[SessionKeys.MODEL_PIPES] = pipes
        st.success("Training complete.")

    if SessionKeys.MODEL_RESULTS not in st.session_state:
        st.info("Run training to see results.")
        st.stop()

    res_df = pd.DataFrame(st.session_state[SessionKeys.MODEL_RESULTS]).sort_values("f1", ascending=False)
    st.subheader("Leaderboard (sorted by F1)")
    st.dataframe(res_df, use_container_width=True)

    best = str(res_df.iloc[0]["model"])
    st.caption(f"Current best: **{best}**")

    pick = st.selectbox("Model to export", options=models, index=models.index(best) if best in models else 0)
    cols = st.columns([1, 2])
    with cols[0]:
        if st.button("Save model bundle", type="secondary", use_container_width=True):
            bundle = st.session_state[SessionKeys.MODEL_PIPES][pick]
            meta = {
                "model_name": pick,
                "metrics": bundle["metrics"],
                "cleaning_config": st.session_state.get(SessionKeys.CLEANING_CFG).__dict__
                if st.session_state.get(SessionKeys.CLEANING_CFG)
                else None,
                "numeric_cols": num_cols,
                "categorical_cols": cat_cols,
            }
            save_artifact(ARTIFACT_PATH, pipeline=bundle["pipeline"], metadata=meta)
            st.success(f"Saved to: `{ARTIFACT_PATH}`")
    with cols[1]:
        with st.expander("Classification report (best)", expanded=False):
            st.text(st.session_state[SessionKeys.MODEL_PIPES][best]["metrics"]["classification_report"])


elif section == "Predict":
    st.subheader("Prediction")
    st.caption("Predict risk for a new client using a saved artifact or an in-session model.")

    clean_df, _ = compute_cleaned(df)
    num_cols, cat_cols = get_feature_columns(clean_df)

    pipe = None
    meta: dict[str, object] = {}
    numeric_ranges: dict[str, dict[str, float]] = {}

    if ARTIFACT_PATH.exists():
        try:
            loaded = load_artifact(ARTIFACT_PATH)
            pipe = loaded["pipeline"]
            meta = loaded.get("metadata", {}) or {}
            numeric_ranges = (meta.get("metrics") or {}).get("numeric_ranges", {})  # type: ignore[assignment]
            st.success(f"Loaded saved artifact: `{ARTIFACT_PATH}`")
        except Exception as e:
            st.warning(f"Could not load artifact: {e}")

    if pipe is None:
        if SessionKeys.MODEL_PIPES not in st.session_state or SessionKeys.MODEL_RESULTS not in st.session_state:
            st.info("Train a model first (Train section), or place an artifact in artifacts/.")
            st.stop()
        res_df = pd.DataFrame(st.session_state[SessionKeys.MODEL_RESULTS]).sort_values("f1", ascending=False)
        best = str(res_df.iloc[0]["model"])
        pipe = st.session_state[SessionKeys.MODEL_PIPES][best]["pipeline"]
        numeric_ranges = st.session_state[SessionKeys.MODEL_PIPES][best]["metrics"].get("numeric_ranges", {})
        st.info(f"Using in-session model: **{best}**")

    numeric_for_form = meta.get("numeric_cols", num_cols) if isinstance(meta, dict) else num_cols
    categorical_for_form = meta.get("categorical_cols", cat_cols) if isinstance(meta, dict) else cat_cols

    def _default_numeric(col: str) -> float:
        if col in clean_df.columns:
            return float(pd.to_numeric(clean_df[col], errors="coerce").median())
        return 0.0

    with st.form("predict_form_v2", border=True):
        st.markdown("**Client features**")
        n1, n2 = st.columns(2)
        payload: dict[str, object] = {}

        with n1:
            st.caption("Numerical inputs")
            for col in list(numeric_for_form):
                val = st.number_input(col, value=float(_default_numeric(col)))
                payload[col] = float(val)
                if col in numeric_ranges:
                    r = numeric_ranges[col]
                    if float(val) < r["min"] or float(val) > r["max"]:
                        st.caption(f"`{col}` outside training range [{r['min']:.3g}, {r['max']:.3g}]")

        with n2:
            st.caption("Categorical inputs")
            for col in list(categorical_for_form):
                options = (
                    sorted(clean_df[col].dropna().astype(str).unique().tolist()) if col in clean_df.columns else []
                ) or ["UNKNOWN"]
                payload[col] = st.selectbox(col, options=options, index=0)

        submit = st.form_submit_button("Run prediction", type="primary")

    if submit:
        x = pd.DataFrame([payload])
        try:
            pred = pipe.predict(x)[0]
            proba = pipe.predict_proba(x)[0] if hasattr(pipe, "predict_proba") else None

            if isinstance(pred, (int, float)) and int(pred) in (0, 1):
                label = "Risque Faible" if int(pred) == 1 else "Risque Elevé"
            else:
                label = str(pred)

            st.subheader("Result")
            st.metric("Prediction", label)

            if proba is not None and len(proba) >= 2:
                out = (
                    pd.DataFrame(
                        {"Class": ["Risque Elevé", "Risque Faible"], "Probability": [float(proba[0]), float(proba[1])]}
                    )
                    .sort_values("Probability", ascending=False)
                    .reset_index(drop=True)
                )
                out["Probability"] = out["Probability"].round(3)
                st.dataframe(out, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

