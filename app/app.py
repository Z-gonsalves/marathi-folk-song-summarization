import sys
import base64
from pathlib import Path

# Add project root folder
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
from transformers import pipeline
from models.emotion_analyzer import detect_emotions


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Marathi Folk Song Summarization & Emotion Analysis",
    layout="wide"
)


# ---------------- LOAD CSS & EMBED LOCAL FONTS ----------------

@st.cache_data
def get_custom_css():
    """Load styles.css and embed the local Noto Sans Devanagari font as base64."""
    css_content = ""
    css_path = ROOT_DIR / "app" / "styles.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            css_content = f.read()

    font_path = ROOT_DIR / "fonts" / "NotoSansDevanagari-VariableFont_wdth,wght.ttf"
    if font_path.exists():
        b64_font = base64.b64encode(font_path.read_bytes()).decode("utf-8")
        font_face = f"""
        @font-face {{
            font-family: 'Noto Sans Devanagari';
            src: url('data:font/truetype;charset=utf-8;base64,{b64_font}') format('truetype');
            font-weight: 300 800;
            font-style: normal;
            font-display: swap;
        }}
        """
        css_content = font_face + "\n" + css_content

    return css_content


st.markdown(
    f"<style>{get_custom_css()}</style>",
    unsafe_allow_html=True
)


# ---------------- HEADER ----------------

st.title("Marathi Folk Song Summarization and Emotion Analysis")
st.caption(
    "A natural language processing system for abstractive summarization of Marathi folk songs "
    "using mT5 and rule-based emotion analysis."
)
st.markdown("---")


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


# ---------------- DASHBOARD METRICS ----------------

m1, m2, m3, m4 = st.columns(4)

m1.metric("Total Songs", f"{len(df):,}")
m2.metric("Genres", df["Genre"].nunique())
m3.metric("Regions", df["Region"].nunique())
m4.metric("Avg. Lyrics Length", f"{int(df['Lyrics'].str.split().str.len().mean())} words")

st.markdown("---")


# ---------------- SIDEBAR ----------------

st.sidebar.subheader("Filters")

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

st.sidebar.caption(f"Showing {len(filtered_df)} of {len(df)} songs.")
st.sidebar.divider()

st.sidebar.subheader("System Info")
st.sidebar.text(
    "Model: mT5-Multilingual-XLSum\n"
    "Baseline: Extractive Frequency\n"
    "Target: Marathi (Devanagari)\n"
    "Corpus: 530+ Folk Songs"
)


# ---------------- SONG SELECTION & OVERVIEW ----------------

song = st.selectbox(
    "Select Song",
    filtered_df["Title"].tolist()
)

row = filtered_df[
    filtered_df["Title"] == song
].iloc[0]

is_evaluated = False
if not evaluation_results.empty and "Title" in evaluation_results.columns:
    is_evaluated = song in evaluation_results["Title"].dropna().tolist()

song_lyrics = str(row["Lyrics"])
song_word_count = len(song_lyrics.split())

st.markdown(
    f"""
    <div class="song-meta-box">
        <div class="song-meta-title">{row['Title']}</div>
        <div class="song-meta-details">
            <span><strong>Genre:</strong> {row['Genre']}</span>
            <span><strong>Region:</strong> {row['Region']}</span>
            <span><strong>Length:</strong> {song_word_count} words</span>
            <span><strong>Evaluation Set:</strong> {"Yes (20-song benchmark)" if is_evaluated else "No (Corpus only)"}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- LOAD SUMMARIZATION MODEL ----------------

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="csebuetnlp/mT5_multilingual_XLSum"
    )


summarizer = load_model()


# ---------------- SESSION STATE MANAGEMENT ----------------

if "current_song" not in st.session_state:
    st.session_state["current_song"] = song
elif st.session_state["current_song"] != song:
    st.session_state["current_song"] = song
    if "summary" in st.session_state:
        del st.session_state["summary"]


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
# TAB 1: SONG INFORMATION
# =========================================================

with tab1:

    left, right = st.columns(2)

    with left:
        st.markdown("<div class='info-field-label'>Genre</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-field-value'>{row['Genre']}</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='info-field-label'>Region</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-field-value'>{row['Region']}</div>", unsafe_allow_html=True)

    st.markdown("<div class='info-field-label' style='margin-top: 0.5rem;'>Historical and Cultural Background</div>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="history-text">{row["History"]}</div>',
        unsafe_allow_html=True
    )


# =========================================================
# TAB 2: LYRICS
# =========================================================

with tab2:

    lyrics_lines = len(song_lyrics.strip().split("\n"))

    st.caption(f"Word count: {song_word_count} | Lines: {lyrics_lines} | Script: Devanagari")

    st.text_area(
        "Original Lyrics",
        song_lyrics,
        height=420,
        label_visibility="visible"
    )


# =========================================================
# TAB 3: SUMMARY
# =========================================================

with tab3:

    st.caption("Model: csebuetnlp/mT5_multilingual_XLSum | Decoding: Beam Search (beams=4, max_length=80, min_length=25)")

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

        summary_text = st.session_state["summary"]
        summary_words = len(summary_text.split())
        compression_ratio = round((1 - (summary_words / max(song_word_count, 1))) * 100)

        st.markdown("### AI Summary")
        st.markdown(
            f"""
            <div class="summary-box">
                <div class="summary-text">{summary_text}</div>
                <div class="summary-meta">
                    Original: {song_word_count} words &nbsp;|&nbsp;
                    Summary: {summary_words} words &nbsp;|&nbsp;
                    Compression: {compression_ratio}% reduction
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "Click 'Generate Summary' to produce an AI-generated summary of the selected song."
        )


# =========================================================
# TAB 4: EMOTION ANALYSIS
# =========================================================

with tab4:

    primary, secondary, confidence = detect_emotions(
        row["Processed_Lyrics"]
    )

    EMOTION_MARATHI = {
        "Heroism": "वीर रस (शौर्य)",
        "Patriotism": "देशभक्ती",
        "Devotion": "भक्ती भाव",
        "Love": "शृंगार रस",
        "Joy": "आनंद / उल्हास",
        "Sadness": "करुण रस (दुःख)",
        "Anger": "रौद्र रस (क्रोध)",
        "Neutral": "तटस्थ"
    }

    st.subheader("Emotion Analysis Results")

    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        st.metric("Primary Emotion", primary)
        st.caption(f"Marathi: {EMOTION_MARATHI.get(primary, '-')}")

    with col_e2:
        sec_label = secondary if secondary != "-" else "None"
        st.metric("Secondary Emotion", sec_label)
        sec_sub = EMOTION_MARATHI.get(secondary, "Single dominant emotion") if secondary != "-" else "Single dominant emotion"
        st.caption(f"Marathi: {sec_sub}")

    with col_e3:
        st.metric("Confidence", f"{confidence}%")
        st.progress(min(confidence / 100.0, 1.0))

    with st.expander("Emotion Keyword Categories Reference"):
        st.markdown(
            """
            | Emotion | Marathi Category | Representative Keywords |
            | :--- | :--- | :--- |
            | Heroism | वीर रस | शिवाजी, वीर, शहाजी, तानाजी, किल्ला, सरदार, तलवार, स्वराज्य, मावळे, युद्ध |
            | Patriotism | देशभक्ती | महाराज, देश, राजा, मराठा, काँग्रेस, ध्वज, भूमी, मातृभूमी |
            | Devotion | भक्ती भाव | विठ्ठल, हरि, राम, देव, देवा, नाम, तुका, मुक्ताई, पांडुरंग, कृष्ण, भक्त |
            | Love | शृंगार रस | जीव, चांदणी, पोरी, बाई, माझ्या, तुला, रूप, मदन |
            | Joy | आनंद | आनंद, गाऊ, खेळ, नाच, उत्सव, सुख, गजर |
            | Sadness | करुण रस | दुःख, रड, विरह, एकटा, गेली, नको, अश्रू |
            | Anger | रौद्र रस | क्रोध, राग, शत्रू, लढ, मोड, वैर, युद्ध |
            """
        )


# =========================================================
# TAB 5: EVALUATION
# =========================================================

with tab5:

    st.subheader("Summarization Evaluation")

    st.markdown(
        "Comparison of the Extractive baseline and mT5 abstractive summarization "
        "using ROUGE metrics evaluated on 20 reference folk songs."
    )

    if not evaluation_scores.empty:

        score_table = evaluation_scores.pivot(
            index="Metric",
            columns="Method",
            values="Average F1 Score"
        )

        score_table = score_table.reindex(
            ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
        )

        st.subheader("Average ROUGE Scores")

        m1_col, m2_col, mL_col = st.columns(3)

        for metric, col in zip(["ROUGE-1", "ROUGE-2", "ROUGE-L"], [m1_col, m2_col, mL_col]):
            with col:
                if metric in score_table.index:
                    ext_val = (
                        score_table.loc[metric, "Extractive"]
                        if "Extractive" in score_table.columns else 0.0
                    )
                    mt5_val = (
                        score_table.loc[metric, "mT5"]
                        if "mT5" in score_table.columns else 0.0
                    )
                    diff = mt5_val - ext_val

                    st.markdown(
                        f"""
                        <div class="eval-card">
                            <div class="eval-card-header">{metric}</div>
                            <div class="eval-card-body">
                                <div><span class="eval-label">Extractive:</span> <strong>{ext_val:.4f}</strong></div>
                                <div><span class="eval-label">mT5:</span> <strong>{mt5_val:.4f}</strong></div>
                            </div>
                            <div class="eval-card-footer">Difference: {diff:+.4f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.caption(
            "Note on ROUGE scores: Extractive summarization selects verbatim sentences directly from "
            "the source lyrics, naturally achieving higher n-gram overlap. The mT5 model generates novel, "
            "paraphrased sentences in Marathi, which results in lower lexical overlap scores while providing true abstractive summarization."
        )

        st.subheader("Method Comparison")

        chart_data = score_table.copy()

        st.bar_chart(
            chart_data,
            y_label="Average F1 Score",
            x_label="ROUGE Metric"
        )

        st.subheader("Detailed Score Table")

        disp_scores = evaluation_scores.copy()
        disp_scores["Average F1 Score"] = (
            disp_scores["Average F1 Score"].round(4)
        )

        st.dataframe(
            disp_scores,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Evaluation scores were not found. Run "
            "'python -P evaluation/run_evaluation.py' "
            "from the project root to generate them."
        )

    st.markdown("---")

    st.subheader("Song-wise Evaluation Results")

    if not evaluation_results.empty:

        if "Title" in evaluation_results.columns:

            available_titles = evaluation_results["Title"].dropna().tolist()

            if song in available_titles:
                selected_result = evaluation_results[
                    evaluation_results["Title"] == song
                ].iloc[0]

                st.markdown(f"**Qualitative Comparison for: {song}**")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown(
                        f"""
                        <div class="qual-box">
                            <div class="qual-title">Reference Summary</div>
                            <div class="qual-content">{selected_result.get('Reference_Summary', 'N/A')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c2:
                    e1 = selected_result.get('Extractive_ROUGE-1', None)
                    e2 = selected_result.get('Extractive_ROUGE-2', None)
                    el = selected_result.get('Extractive_ROUGE-L', None)
                    score_str = (
                        f"R-1: {e1:.4f} | R-2: {e2:.4f} | R-L: {el:.4f}"
                        if pd.notna(e1) else ""
                    )
                    st.markdown(
                        f"""
                        <div class="qual-box">
                            <div class="qual-title">Extractive Baseline</div>
                            <div class="qual-content">{selected_result.get('Extractive_Summary', 'N/A')}</div>
                            <div class="qual-meta">{score_str}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c3:
                    m1 = selected_result.get('mT5_ROUGE-1', None)
                    m2 = selected_result.get('mT5_ROUGE-2', None)
                    ml = selected_result.get('mT5_ROUGE-L', None)
                    m_score_str = (
                        f"R-1: {m1:.4f} | R-2: {m2:.4f} | R-L: {ml:.4f}"
                        if pd.notna(m1) else ""
                    )
                    st.markdown(
                        f"""
                        <div class="qual-box">
                            <div class="qual-title">mT5 Abstractive</div>
                            <div class="qual-content">{selected_result.get('mT5_Summary', 'N/A')}</div>
                            <div class="qual-meta">{m_score_str}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with st.expander("View Numerical Table for Selected Song"):
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
                        if col in selected_result.index
                    ]
                    st.dataframe(
                        pd.DataFrame([selected_result[summary_columns]]),
                        use_container_width=True,
                        hide_index=True
                    )

            else:
                st.info(
                    "The selected song is part of the broader corpus and was not included in the 20-song reference evaluation set."
                )

            with st.expander("View Complete 20-Song Benchmark Dataset"):
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