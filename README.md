# AI PDF Chatbot

An interactive, locally running AI chatbot that lets users upload PDF documents and ask questions about their contents. It combines semantic search with a local Ollama language model to generate context-aware answers.

The application runs on your own computer and does not require a hosted LLM API.

## Features

- Upload and process PDF documents
- Extract text using PyMuPDF
- Split extracted text into overlapping chunks
- Generate semantic embeddings using Sentence Transformers
- Retrieve relevant passages using FAISS similarity search
- Generate answers using the local `qwen2.5:3b` model through Ollama
- Display relevant source passages and page references
- Interactive Streamlit interface with chat history and suggested questions
- Automated unit tests for the retrieval pipeline

## Technology Stack

- Python 3.11
- Streamlit
- PyMuPDF
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- Ollama (`qwen2.5:3b`)
- NumPy
- Python `unittest`
- Git and GitHub

## Project Structure

```text
AI-PDF-Chatbot/
├── .streamlit/
│   └── config.toml
├── data/
├── tests/
│   └── test_retrieval.py
├── app.py
├── retrieval.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `.venv/` virtual environment and locally uploaded documents are not intended to be committed to GitHub.

## Prerequisites

Before running the application, install:

- Python 3.11
- Ollama
- Git (optional, for cloning the repository)

The application also requires the `qwen2.5:3b` model to be available in Ollama.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/ahsanhabibabir04/AI-PDF-Chatbot.git
cd AI-PDF-Chatbot
```

Alternatively, download the repository as a ZIP file from GitHub and extract it.

### 2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run this command in the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

### 4. Install Python dependencies

Upgrade pip and install the required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The embedding model may be downloaded the first time it is loaded. An internet connection may be required for the initial download.

## Ollama Setup

### 1. Install Ollama

Download and install Ollama from:

https://ollama.com/download

### 2. Download the language model

Open PowerShell and run:

```powershell
ollama pull qwen2.5:3b
```

### 3. Verify the model

```powershell
ollama list
```

Confirm that `qwen2.5:3b` appears in the model list.

Ollama must be running locally while you use the chatbot.

## Run the Application

From the project directory, activate the virtual environment if it is not already active:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start the application:

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL, usually:

http://localhost:8501

Open that URL in your browser, upload a PDF, and start asking questions about its content.

Keep Ollama running while using the application.

## Run Automated Tests

Run the retrieval unit tests from the project root:

```powershell
python -m unittest discover -s tests -v
```

The tests cover text chunking, index construction, and retrieval behavior. The retrieval tests use a fake embedding model where appropriate, so the tested functions do not need to download a real embedding model for every test.

## How It Works

1. **PDF ingestion:** Extract text from an uploaded PDF.
2. **Text chunking:** Divide the extracted text into overlapping chunks.
3. **Embedding generation:** Convert chunks into numerical embeddings using Sentence Transformers.
4. **Vector indexing:** Store embeddings in a FAISS similarity-search index.
5. **Question retrieval:** Find passages relevant to the user's question.
6. **Answer generation:** Send the retrieved context to the local Ollama model.
7. **Source display:** Show relevant passages and page references when available.

## Troubleshooting

### Ollama connection error

- Ensure Ollama is installed and running.
- Run `ollama list` to verify that the model is available.
- Run `ollama run qwen2.5:3b` to check whether the model starts successfully.
- Restart Streamlit if necessary.

### Missing Python module

Activate the virtual environment and install dependencies again:

```powershell
python -m pip install -r requirements.txt
```

### Embedding model fails to load

Check your internet connection for the initial download. Also ensure that sufficient RAM and disk space are available.

### PDF text cannot be extracted

Scanned PDFs may contain images instead of selectable text. OCR support may be needed for those documents.

### Slow responses

Performance depends on your hardware, PDF size, embedding generation, and local language-model inference. Initial model loading may take longer.

## Limitations

- This is a local application, not a publicly hosted service.
- Other users must install the dependencies and Ollama model on their own computers.
- Scanned PDFs may require OCR.
- Answers may be incorrect or unsupported by the document.
- Source passages and page references should be checked against the original PDF.
- Local inference speed depends on available hardware.
- No hosted LLM API key is required for local Ollama inference.

## Privacy

The application is designed for local use. Uploading the source code to GitHub does not automatically publish PDFs uploaded while running the application locally.

Review application behavior and dependencies before processing confidential documents. Internet access may be needed to install packages and download models.

## Future Improvements

- Add OCR for scanned PDFs
- Support multiple PDF documents
- Improve retrieval quality and answer grounding
- Add retrieval evaluation
- Support configurable local Ollama models
- Improve error handling and document-processing feedback

## Author

**M.A. Ahsan Habib Abir**

GitHub: [ahsanhabibabir04](https://github.com/ahsanhabibabir04)

## License

No license has been specified. Unless a license is added, others should not assume they have permission to reuse, modify, or redistribute this project's code.
