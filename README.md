# Document RAG Assistant

A Streamlit application for Retrieval-Augmented Generation (RAG) with document uploads. This application allows users to upload documents, ask questions about them, and get AI-powered responses based on document content.

## Features

- **Document Processing**: Upload and process PDF, DOCX, TXT, and CSV files
- **Multiple LLM Support**: Configurable OpenAI model selection
- **Embedding Models**: Choose from various OpenAI embedding models
- **Customizable Chunking**: Adjust chunk size and overlap parameters
- **Flexible Retrieval**: Select different similarity metrics and control retrieval parameters
- **Comparison Mode**: Compare RAG responses with direct LLM responses
- **Responsive UI**: User-friendly interface with conversation history

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shatzzzy/rag-app.git
cd rag-app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
   - Create a file named `config.py` with the following content:
   ```python
   api_key = "your_openai_api_key"  # Replace with your API key or leave as None to input in the UI
   ```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Start the application and enter your OpenAI API key in the sidebar if not provided in config.py
2. Upload a document (PDF, DOCX, TXT, or CSV)
3. Adjust the model settings, chunking parameters, and retrieval settings as needed
4. Ask questions about your document in the chat interface

## Configuration Options

- **Model Settings**: Select from various OpenAI models and embedding models
- **Chunking Settings**: Adjust chunk size and overlap percentage
- **Retrieval Settings**: 
  - Control the number of chunks to retrieve
  - Select the similarity metric (Cosine Similarity, Euclidean Distance, Dot Product)
  - Adjust the temperature for response generation

## System Requirements

- Python 3.8+
- OpenAI API key
- Sufficient memory to handle document processing (requirements vary based on document size)

## Dependencies

Key dependencies include:
- streamlit
- openai
- langchain
- pandas
- matplotlib
- scikit-learn
- and others (see requirements.txt for the complete list)

## License

[MIT License](LICENSE)

## Acknowledgments

This application uses the following key libraries:
- [LangChain](https://github.com/hwchase17/langchain)
- [OpenAI API](https://openai.com/)
- [Streamlit](https://streamlit.io/)
