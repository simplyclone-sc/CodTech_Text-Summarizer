import streamlit as st
from transformers import pipeline

# Load summarization pipeline with caching
@st.cache_resource(show_spinner=True)
def initialize_summarizer():
    summarization_model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return summarization_model

summarize_text = initialize_summarizer()

st.set_page_config(page_title="Text Summarizer", page_icon="📝")
st.header("📝 Text Summarizer App")
st.markdown("Use this app to condense lengthy text into short summaries using a powerful NLP model.")

# Default content
preset_content = (
    "Artificial Intelligence (AI) is revolutionizing the modern world by introducing automation into processes "
    "that were previously handled manually. From healthcare and education to transportation and finance, AI is "
    "enabling more accurate predictions, smarter decision-making, and improved efficiencies. In particular, "
    "Natural Language Processing (NLP), a subfield of AI, is allowing machines to understand and generate human "
    "language. Applications like text summarization, sentiment analysis, and machine translation are becoming "
    "increasingly common in both consumer and enterprise products. As AI continues to evolve, it is crucial to "
    "ensure its development is guided by ethical principles to maximize benefits and minimize risks to society."
)

# Text input field
user_input = st.text_area("Input Text", value=preset_content, height=250)

# Auto-summarize default content
if user_input == preset_content:
    with st.spinner("Creating summary..."):
        result = summarize_text(user_input, max_length=70, min_length=30, do_sample=False)
    st.subheader("📌 Auto Summary")
    st.success(result[0]['summary_text'])

# Button-based summarization
if st.button("Generate Summary"):
    if user_input.strip():
        with st.spinner("Summarizing your text..."):
            result = summarize_text(user_input, max_length=70, min_length=30, do_sample=False)
        st.subheader("📌 Generated Summary")
        st.success(result[0]['summary_text'])
    else:
        st.warning("❗ Please input some text to summarize.")