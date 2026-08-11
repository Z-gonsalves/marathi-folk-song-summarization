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

st.write(
    "Generate summaries and analyze emotions from Marathi folk songs."
)


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_csv(
        ROOT_DIR / "outputs" / "final_preprocessed_dataset.csv",
        encoding="utf-8"
    )


df = load_data()


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

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Song Information",
        "Lyrics",
        "Summary",
        "Emotion Analysis"
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