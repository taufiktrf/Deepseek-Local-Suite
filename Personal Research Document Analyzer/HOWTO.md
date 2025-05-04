## Getting Started

### Prerequisites

- Ensure you have [Ollama](https://ollama.ai/) installed on your system

### Running the Application

1. Start the DeepSeek model with Ollama:

```bash
ollama run deepseek-r1:1.5b
```

2. Launch the research assistant interface:

```bash
streamlit run research_assistant.py
```

### Configuration

To use a different language model, modify the model name in the `ResearchAssistant` class within the `research_assistant.py` file.
