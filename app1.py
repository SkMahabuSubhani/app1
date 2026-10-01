import streamlit as st
from transformers import pipeline
import re
import io

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Briefly AI - Text Summarizer",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 30px;
}

.summary-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #ffffff;
    border: 1px solid #ddd;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="title">🧠 Briefly AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Turn large text into a short, meaningful summary</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# LOAD AI MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    return summarizer


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):

    # Remove unnecessary spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# ---------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------

def split_text(text, max_words=450):

    words = text.split()

    chunks = []

    for i in range(0, len(words), max_words):

        chunk = " ".join(
            words[i:i + max_words]
        )

        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------
# SUMMARIZE TEXT
# ---------------------------------------------------------

def summarize_text(text, summarizer):

    text = clean_text(text)

    if len(text.split()) < 30:

        return text

    chunks = split_text(text)

    chunk_summaries = []

    progress = st.progress(0)

    for index, chunk in enumerate(chunks):

        try:

            result = summarizer(
                chunk,
                max_length=120,
                min_length=30,
                do_sample=False
            )

            summary = result[0]["summary_text"]

            chunk_summaries.append(summary)

        except Exception as e:

            st.warning(
                f"Could not summarize chunk {index + 1}: {e}"
            )

        progress.progress(
            (index + 1) / len(chunks)
        )

    # Combine chunk summaries

    combined_summary = " ".join(
        chunk_summaries
    )

    # If multiple chunks exist,
    # summarize them again

    if len(chunk_summaries) > 1:

        try:

            final_result = summarizer(
                combined_summary,
                max_length=180,
                min_length=60,
                do_sample=False
            )

            return final_result[0]["summary_text"]

        except Exception:

            return combined_summary

    return combined_summary


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

st.subheader("📄 Enter Large Text")

text_input = st.text_area(
    "Paste your text here:",
    height=300,
    placeholder="Paste an article, research paper, notes, report, etc..."
)

st.write("OR")

uploaded_file = st.file_uploader(
    "Upload a TXT file",
    type=["txt"]
)

# ---------------------------------------------------------
# GET INPUT
# ---------------------------------------------------------

text = ""

if uploaded_file is not None:

    text = uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )

elif text_input:

    text = text_input


# ---------------------------------------------------------
# TEXT INFORMATION
# ---------------------------------------------------------

if text:

    word_count = len(text.split())

    character_count = len(text)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Words",
            word_count
        )

    with col2:
        st.metric(
            "Characters",
            character_count
        )

    with col3:
        st.metric(
            "Estimated chunks",
            (word_count // 450) + 1
        )


# ---------------------------------------------------------
# SUMMARIZE BUTTON
# ---------------------------------------------------------

if st.button(
    "✨ Generate Brief Summary",
    use_container_width=True
):

    if not text:

        st.error(
            "Please enter text or upload a TXT file."
        )

    else:

        with st.spinner(
            "AI is analyzing your text..."
        ):

            try:

                summarizer = load_model()

                summary = summarize_text(
                    text,
                    summarizer
                )

                st.session_state["summary"] = summary

            except Exception as e:

                st.error(
                    f"Error: {e}"
                )


# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

if "summary" in st.session_state:

    st.markdown("---")

    st.subheader("📝 Brief Summary")

    st.markdown(
        f"""
        <div class="summary-box">
        {st.session_state["summary"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # SUMMARY STATISTICS
    # -----------------------------------------------------

    original_words = len(text.split())

    summary_words = len(
        st.session_state["summary"].split()
    )

    reduction = 0

    if original_words > 0:

        reduction = (
            1 - summary_words / original_words
        ) * 100

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Original Words",
            original_words
        )

    with col2:

        st.metric(
            "Summary Words",
            summary_words
        )

    st.info(
        f"📉 Approximately {reduction:.1f}% of the original text was removed."
    )

    # -----------------------------------------------------
    # DOWNLOAD SUMMARY
    # -----------------------------------------------------

    summary_file = io.BytesIO(
        st.session_state["summary"].encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download Summary",
        data=summary_file,
        file_name="brief_summary.txt",
        mime="text/plain"
    )
