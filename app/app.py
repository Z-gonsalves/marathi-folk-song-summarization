
import sys
from pathlib import Path

# Add project root folder
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
from transformers import pipeline
from models.emotion_analyzer import detect_emotions


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Marathi Folk Songs Summarization & Emotion Analysis",
    page_icon="🎵",
    layout="wide"
)


# ---------------- LOAD CSS ----------------

with open(ROOT_DIR / "app" / "styles.css", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# ---------------- HEADER ----------------

st.title("Marathi Folk Songs Summarization & Emotion Analysis")

st.markdown(
    "<div class='app-subtitle'>Generate summaries and analyze emotions from Marathi folk songs.</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='report-header'><span class='report-kicker'>Folk Song Analysis Report</span></div>",
    unsafe_allow_html=True
)


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_csv(
        ROOT_DIR / "outputs" / "final_preprocessed_dataset.csv",
        encoding="utf-8"
    )


@st.cache_data
def load_evaluation_scores():
    path = ROOT_DIR / "evaluation" / "rouge_scores.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_evaluation_results():
    path = ROOT_DIR / "evaluation" / "evaluation_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


df = load_data()
evaluation_scores = load_evaluation_scores()
evaluation_results = load_evaluation_results()


# ---------------- DASHBOARD ----------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Songs", len(df))

col2.metric(
    "Genres",
    df["Genre"].nunique()
)

col3.metric(
    "Regions",
    df["Region"].nunique()
)

col4.metric(
    "Avg. Lyrics Length",
    int(df["Lyrics"].str.split().str.len().mean())
)

st.divider()


# ---------------- SIDEBAR ----------------

st.sidebar.title("Filters")

genre = st.sidebar.selectbox(
    "Genre",
    ["All"] + sorted(df["Genre"].dropna().unique().tolist())
)

if genre != "All":
    filtered_df = df[df["Genre"] == genre]
else:
    filtered_df = df.copy()


region = st.sidebar.selectbox(
    "Region",
    ["All"] + sorted(filtered_df["Region"].dropna().unique().tolist())
)

if region != "All":
    filtered_df = filtered_df[
        filtered_df["Region"] == region
    ]


# ---------------- SONG SELECTION ----------------

song = st.selectbox(
    "Select Song",
    filtered_df["Title"].tolist()
)

row = filtered_df[
    filtered_df["Title"] == song
].iloc[0]


# ---------------- LOAD SUMMARIZATION MODEL ----------------

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="csebuetnlp/mT5_multilingual_XLSum"
    )


summarizer = load_model()


# ---------------- TABS ----------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Song Information",
        "Lyrics",
        "Summary",
        "Emotion Analysis",
        "Evaluation"
    ]
)


# =========================================================
# SONG INFORMATION TAB
# =========================================================

with tab1:

    left, right = st.columns(2)

    with left:
        st.subheader("Genre")
        st.write(row["Genre"])

    with right:
        st.subheader("Region")
        st.write(row["Region"])

    st.subheader("History")
    st.write(row["History"])


# =========================================================
# LYRICS TAB
# =========================================================

with tab2:

    st.subheader("Original Lyrics")

    st.text_area(
        "Lyrics",
        row["Lyrics"],
        height=400,
        label_visibility="collapsed"
    )


# =========================================================
# SUMMARY TAB
# =========================================================

with tab3:

    if st.button(
        "Generate Summary",
        key="summary_button"
    ):

        with st.spinner("Generating summary..."):

            summary = summarizer(
                row["Processed_Lyrics"],
                max_length=80,
                min_length=25,
                do_sample=False
            )

        st.session_state["summary"] = (
            summary[0]["summary_text"]
        )

    if "summary" in st.session_state:

        st.subheader("AI Summary")

        st.markdown(
            f"""
            <div class="summary-card">
                {st.session_state["summary"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "Click 'Generate Summary' to generate an AI-based summary."
        )


# =========================================================
# EMOTION ANALYSIS TAB
# =========================================================

with tab4:

    primary, secondary, confidence = detect_emotions(
        row["Processed_Lyrics"]
    )

    st.subheader("Emotion Analysis")

    emotion_col1, emotion_col2, emotion_col3 = st.columns(3)

    with emotion_col1:

        st.markdown(
            f"""
            <div class="emotion-card">
                <h4>Primary Emotion</h4>
                <h2>{primary}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with emotion_col2:

        st.markdown(
            f"""
            <div class="emotion-card">
                <h4>Secondary Emotion</h4>
                <h2>{secondary}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with emotion_col3:

        st.markdown(
            f"""
            <div class="emotion-card">
                <h4>Confidence</h4>
                <h2>{confidence}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# EVALUATION TAB
# =========================================================

with tab5:

    st.subheader("Summarization Evaluation")

    st.markdown(
        "Comparison of the Extractive and mT5 summarization methods "
        "using ROUGE evaluation."
    )

    if not evaluation_scores.empty:

        # Prepare average ROUGE score table
        score_table = evaluation_scores.pivot(
            index="Metric",
            columns="Method",
            values="Average F1 Score"
        )

        score_table = score_table.reindex(
            ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
        )

        st.subheader("Average ROUGE Scores")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        for metric, column in zip(
            ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
            [metric_col1, metric_col2, metric_col3]
        ):
            with column:
                if metric in score_table.index:
                    extractive_score = score_table.loc[
                        metric, "Extractive"
                    ] if "Extractive" in score_table.columns else None

                    mt5_score = score_table.loc[
                        metric, "mT5"
                    ] if "mT5" in score_table.columns else None

                    st.metric(
                        metric,
                        f"{extractive_score:.4f}"
                        if pd.notna(extractive_score) else "N/A",
                        delta=(
                            f"{mt5_score - extractive_score:+.4f} mT5 vs Extractive"
                            if pd.notna(extractive_score)
                            and pd.notna(mt5_score)
                            else None
                        ),
                        delta_color="off"
                    )

        st.subheader("Method Comparison")

        chart_data = score_table.copy()

        st.bar_chart(
            chart_data,
            y_label="Average F1 Score",
            x_label="ROUGE Metric"
        )

        st.caption(
            "Higher ROUGE scores indicate greater word or sequence "
            "overlap with the reference summaries. They do not, by "
            "themselves, establish semantic accuracy or fluency."
        )

        st.subheader("Detailed Score Table")

        display_scores = evaluation_scores.copy()
        display_scores["Average F1 Score"] = (
            display_scores["Average F1 Score"].round(4)
        )

        st.dataframe(
            display_scores,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Evaluation scores were not found. Run "
            "'python -P evaluation/run_evaluation.py' "
            "from the project root to generate them."
        )

    st.divider()

    st.subheader("Song-wise Evaluation Results")

    if not evaluation_results.empty:

        if "Title" in evaluation_results.columns:

            available_titles = evaluation_results["Title"].dropna().tolist()

            if song in available_titles:
                selected_result = evaluation_results[
                    evaluation_results["Title"] == song
                ]

                st.markdown(f"**Selected song: {song}**")

                summary_columns = [
                    col for col in [
                        "Title",
                        "Reference_Summary",
                        "Extractive_Summary",
                        "Extractive_ROUGE-1",
                        "Extractive_ROUGE-2",
                        "Extractive_ROUGE-L",
                        "mT5_Summary",
                        "mT5_ROUGE-1",
                        "mT5_ROUGE-2",
                        "mT5_ROUGE-L"
                    ]
                    if col in selected_result.columns
                ]

                st.dataframe(
                    selected_result[summary_columns],
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.info(
                    "This song is not part of the 20-song evaluation set. "
                    "The average ROUGE scores above are still available."
                )

            with st.expander("View all evaluated songs"):
                st.dataframe(
                    evaluation_results,
                    use_container_width=True,
                    hide_index=True
                )

        else:
            st.warning(
                "The evaluation results file does not contain a Title column."
            )

    else:

        st.info(
            "Song-wise results are not available. Run the evaluation "
            "script to generate evaluation_results.csv."
        )