import streamlit as st
from openai import OpenAI


class LocalCodeAssistant:
    def __init__(self):
        self.client = OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1/",
        )

        self.model = "deepseek-r1:1.5b"

    def process_request(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
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


def get_system_prompts():
    return {
        "Code Generation": """You are an expert Python programmer who creates clean, efficient, and well-documented code.
    When writing code:
    1. Add a brief comment explaining purpose
    2. Write clear docstrings for functions
    3. Choose descriptive variable names
    4. Include comments for complex logic
    5. Follow PEP 8 style guidelines
    6. Show example usage
    7. Handle common edge cases""",
        "Code Explanation": """You are a knowledgeable coding instructor.
    When explaining code:
    1. Describe overall purpose and functionality
    2. Explain each major component
    3. Identify key programming concepts
    4. Detail the execution flow
    5. Clarify important variables and functions
    6. Highlight clever techniques or patterns
    7. Point out educational aspects for learners""",
        "Code Review": """You are a senior code reviewer with Python expertise.
    Review code for:
    1. Potential bugs or logical errors
    2. Performance optimization opportunities
    3. Security vulnerabilities
    4. Style and PEP 8 compliance
    5. Error handling improvements
    6. Documentation completeness
    7. Modularity and reusability
    8. Memory efficiency
    Provide specific improvement suggestions.""",
    }


def get_example_prompts():
    return {
        "Code Generation": """Generate a Python function that calculates the factorial of a number.
    The function should be named `factorial` and take one argument, `n`, which is a non-negative integer.
    Include error handling for negative inputs and provide an example usage.""",
        "Code Explanation": """Explain the following code snippet:
    ```python
    def fibonacci(n):
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        elif n == 2:
            return [0, 1]
        else:
            fib_sequence = [0, 1]
            for i in range(2, n):
                fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
            return fib_sequence
    ```""",
        "Code Review": """Review the following code for best practices and potential improvements:
    ```python
    def add_numbers(a, b):
        return a + b
    ```""",
    }


def main():
    st.set_page_config(
        page_title="DeepSeek R1 Code Assistant", page_icon="🤖", layout="wide"
    )

    st.title("🤖 Local Personal Code Assistant")
    st.markdown(
        """
    Using DeepSeek R1 models running locally through Ollama
    """
    )

    system_prompts = get_system_prompts()
    example_prompts = get_example_prompts()

    # Sidebar
    st.sidebar.title("Settings")
    mode = st.sidebar.selectbox(
        "Choose Mode", ["Code Generation", "Code Explanation", "Code Review"]
    )

    # Show current mode description
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Mode**: {mode}")
    st.sidebar.markdown("**Mode Description:**")
    st.sidebar.markdown(system_prompts[mode].replace("\n", "\n\n"))

    # Main content area
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown(f"### Input for {mode}")
        user_prompt = st.text_area(
            "Enter your prompt or code:",
            height=300,
            placeholder=example_prompts[mode],
            value=example_prompts[mode],
        )

        process_button = st.button(
            "🚀 Process", type="primary", use_container_width=True
        )

    with col2:
        st.markdown("### Output")
        output_container = st.container()

    if process_button:
        if user_prompt:
            with st.spinner("Processing..."):
                with output_container:
                    assistant = LocalCodeAssistant()
                    assistant.process_request(system_prompts[mode], user_prompt)
        else:
            st.warning("⚠️ Please enter some input!")

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center'>
        <p>Made with ❤️ using DeepSeek R1 and Ollama</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
