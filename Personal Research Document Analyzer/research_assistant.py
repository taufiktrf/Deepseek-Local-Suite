import streamlit as st
import PyPDF2
from openai import OpenAI


class ResearchAssistant:
    def __init__(self):
        self.client = OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1/",
        )

        self.model = "deepseek-r1:1.5b"

    def extract_text_from_pdf(self, uploaded_file):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""

        if uploaded_file.type == "application/pdf":
            for page in pdf_reader.pages:
                text += page.extract_text()

        else:
            text = str(uploaded_file.read(), "utf-8")

        return text

    def analyze_content(self, text, query):
        prompt = f"""Analyze this text and answer the following query:
            Text: {text[:2000]}...
            Query: {query}
            
            Provide:
            1. Direct answer to the query
            2. Supporting evidences
            3. Key findings
            4. Limitations of the analysis
            """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research assistant skilled in analyzing academic and technical documents.",
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )

            result = st.empty()
            collected_chunks = []

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    collected_chunks.append(chunk.choices[0].delta.content)
                    result.markdown("".join(collected_chunks))

            return "".join(collected_chunks)

        except Exception as e:
            return f"Error: {str(e)}"


def main():
    st.set_page_config(page_title="Local Research Assistant", layout="wide")
    st.title("Personal Research Document Analyzer")

    assistant = ResearchAssistant()

    # Sidebar for document upload
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload research documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

    # Main content area
    if uploaded_files:
        st.write(f"📎 {len(uploaded_files)} documents uploaded")

        # Query input
        query = st.text_area(
            "What would you like to know about these documents?",
            placeholder="Example: What are the main findings regarding climate change impact?",
            height=100,
        )

        if st.button("Analyze", type="primary"):
            with st.spinner("Analyzing documents..."):
                # Process each document
                for file in uploaded_files:
                    st.write(f"### Analysis of {file.name}")
                    text = assistant.extract_text_from_pdf(file)

                    # Create tabs for different analyses
                    tab1, tab2, tab3 = st.tabs(
                        ["Main Analysis", "Key Points", "Summary"]
                    )

                    with tab1:
                        assistant.analyze_content(text, query)

                    with tab2:
                        assistant.analyze_content(
                            text, "Extract key points and findings"
                        )

                    with tab3:
                        assistant.analyze_content(text, "Provide a brief summary")

                # Compare documents if multiple
                if len(uploaded_files) > 1:
                    st.write("### Cross-Document Analysis")
                    st.write("Comparing findings across documents...")
                    # Add comparison logic here


if __name__ == "__main__":
    main()
