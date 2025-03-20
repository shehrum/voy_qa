# 🤖 Voy Q&A System

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.11-brightgreen)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-lightgrey)](https://openai.com/)

</div>

This project implements a Q&A system for Voy's telehealth services using LLMs and document retrieval. The system leverages retrieval-augmented generation (RAG) to provide accurate, relevant, and safe answers to user questions about Voy's telehealth weight loss services.

## 📊 Project Overview

### Core Approach
The solution employs a Retrieval-Augmented Generation (RAG) architecture with these key components:

1. **Data Acquisition** - Scrapes information from Voy's official knowledge base
2. **Document Processing** - Chunks and indexes content for efficient retrieval  
3. **Semantic Retrieval** - Finds contextually relevant documents for each query
4. **LLM Response Generation** - Creates coherent, accurate answers using retrieved context
5. **Safety Guardrails** - Implements confidence scoring and medical disclaimers

### Technical Stack
- **Data Source**: Zendesk Knowledge Base API
- **Vector Database**: ChromaDB
- **Embedding Model**: OpenAI Embeddings
- **LLM**: GPT-4 Turbo
- **Frontend**: Gradio web interface
- **Evaluation**: Automated testing pipeline with quantitative metrics

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Code Organization](#-code-organization)
- [Setup](#-setup)
- [Usage](#-usage)
- [Scraper Implementation](#-scraper-implementation)
- [Architecture & Design Choices](#-architecture--design-choices)
- [Features](#-features)
- [User Interface](#-user-interface)
- [Safety Measures](#-safety-measures)
- [Evaluation System](#-evaluation-system)
- [Sample Test Questions](#-sample-test-questions)
- [Future Improvements](#-future-improvements)

## 📁 Project Structure

```
voy_qa/
├── src/             # Source code
│   ├── scraper.py   # FAQ scraping functionality
│   ├── qa.py        # Q&A system implementation
│   ├── evaluation.py # Evaluation framework
│   └── app.py       # Gradio web interface
├── tests/           # Test cases
├── evaluate.py      # Evaluation script
├── data/            # Stores scraped FAQ data and embeddings
└── evaluation/      # Stores evaluation results and reports
└── requirements.txt
```
## 🚀 Setup

1. **Install dependencies:**
```bash
conda create --name myenv --file requirements.txt
conda activate myenv
```

3. **Create a `.env` file with your OpenAI API key:**
```
OPENAI_API_KEY=your_key_here
```

## 🔧 Usage

**Run the Scraper:**
```bash
python src/scraper.py
```

**Run the Gradio interface:**
```bash
python src/app.py
```

**Run tests:**
```bash
pytest tests/
```


## 🧩 Code Organization

The codebase follows a modular design with clear separation of concerns:

### 🔄 Core Components

1. **`scraper.py`**: Implements the `ZendeskAPIScraper` class that:
   - Connects to Zendesk Help Center API
   - Retrieves and processes article content
   - Cleans and structures data for storage
   - Saves formatted data to JSON

2. **`qa.py`**: Implements the `VoyQASystem` class that:
   - Loads and indexes document data
   - Manages vector database operations  
   - Retrieves context based on query relevance
   - Generates responses using OpenAI's LLM
   - Assesses confidence levels

3. **`app.py`**: Implements the Gradio web interface that:
   - Creates a user-friendly frontend
   - Formats responses with sources and confidence
   - Includes medical disclaimers
   - Provides pre-populated example questions
   - Handles error cases gracefully

4. **`evaluation.py`**: Implements evaluation tools that:
   - Create test datasets with expected results
   - Run evaluations on system performance
   - Calculate metrics on accuracy and relevance
   - Generate visual and textual reports

### 🔁 Data Flow

The system's data flow follows this sequence:
1. Scraper extracts data → JSON storage
2. QA system loads JSON → Vector database
3. User query → Context retrieval → LLM response generation
4. Response formatting → UI presentation

### 📏 Design Principles

The implementation adheres to:
- **Modularity**: Components with single responsibilities
- **Extensibility**: Easy integration of new data sources or models
- **Safety**: Multiple layers of confidence checking and disclaimers
- **Transparency**: Clear citation of sources in all responses

## 📚 Scraper Implementation

The VoyQA system uses a Zendesk API scraper to collect FAQ data from Voy's help center. 

### 🔄 Scraping Process

The scraper follows these steps:

1. **Connect to Zendesk API** - Accesses Voy's help center via public API endpoints
2. **Fetch Article Index** - Retrieves metadata for all published articles with pagination
3. **Process Each Article** - Fetches full content for each article
4. **Clean Data** - Removes HTML tags and extracts key information (title, content, URL)
5. **Structure Output** - Formats data as JSON with consistent structure
6. **Save Results** - Stores processed data in `data/faq_data.json`

### 🔌 Integration with QA System

1. Scraper runs independently to collect fresh data
2. QA system loads the JSON during initialization
3. Documents are chunked, embedded, and indexed for retrieval

Run the scraper with:
```bash
python src/scraper.py
```

## 🏗️ Architecture & Design Choices

### RAG System Architecture

The VoyQA system follows a modern retrieval-augmented generation architecture with the following components:

#### 1. 📄 **Document Processing Pipeline**
   - Data is sourced from Voy's FAQ pages and structured as JSON
   - Documents are split into optimal chunks (1000 tokens with 200 token overlap)
   - Document metadata preserves original source URLs for attribution

#### 2. 🔍 **Vector Database & Retrieval**
   - Uses Chroma as the vector database for efficient similarity search
   - OpenAI embeddings create dense vector representations
   - Implements relevance threshold filtering (0.7) to prevent low-quality retrievals
   - Returns multiple sources (k=3) to provide comprehensive context

#### 3. 🧠 **LLM Integration**
   - Uses GPT-4 Turbo with low temperature (0.1) for consistent, factual responses
   - Custom prompt template enforces accuracy, citation, and medical disclaimers
   - Chain-based architecture enables future extension to more complex reasoning flows

#### 4. ⚖️ **Confidence Assessment**
   - Multi-level confidence scoring based on:
     - Presence of relevant retrieved documents
     - Question relevance to Voy's domain
     - Explicit checks for empty or malformed queries
   - Low confidence triggers safe fallback responses

### System Design 

The system was designed with several key principles in mind:

#### 1. ✅ **Accuracy First**
   - Only using information from vetted source documents
   - Setting strict relevance thresholds for retrieval
   - Including source attribution in all responses
   - Clearly expressing uncertainty when information is limited

#### 2. 🏥 **Medical Safety**
   - Including medical disclaimers for treatment-related questions
   - Avoiding inference or extrapolation beyond source material
   - Directing users to healthcare professionals for medical advice
   - Providing clear confidence indicators for all responses

#### 3. 🔌 **Extensibility**
   - Easy addition of new data sources
   - Swapping of embedding or LLM models
   - Integration of more sophisticated retrieval techniques
   - Enhanced evaluation methodologies

## ✨ Features

- 🕸️ Web scraping of Voy FAQ pages
- 📊 Document retrieval using ChromaDB
- 🤖 LLM-powered question answering with OpenAI
- 🖥️ Gradio web interface
- 📈 Evaluation metrics and safety measures

## 🖥️ User Interface

The system features a clean, user-friendly Gradio web interface that:

1. **Simple Question Input** - Text area for entering questions about Voy's services
2. **Formatted Responses** - Clearly structured answers with source citations 
3. **Confidence Indicators** - Visual indicators for answer reliability
4. **Medical Disclaimers** - Clear notices for medical information
5. **Example Questions** - Pre-populated examples to demonstrate capabilities

The interface is designed to be:
- **Accessible** - Clean layout with proper contrast and readability
- **Informative** - Provides metadata about response sources and confidence
- **Cautious** - Clearly labels low-confidence responses with warning indicators
- **Educational** - Includes example questions to guide users

![Gradio Interface Screenshot](path-to-screenshot.png) <!-- Optional: Add a screenshot if available -->

## 🛡️ Safety Measures

#### 1. 📝 **Source Attribution**
All responses include references to source documents, allowing users to verify information.

#### 2. 🎯 **Confidence Scoring**
The system assesses its confidence in each response:
- **High Confidence**: Question is relevant to Voy's domain, multiple high-quality sources retrieved
- **Low Confidence**: Question is out-of-domain, no relevant sources found, or input is malformed

#### 3. ⚕️ **Medical Disclaimer**
Clear disclaimers are included when discussing treatments or medications, emphasizing that the system does not provide medical advice.

#### 4. 🔒 **Hallucination Prevention**
- Strict retrieval thresholds reduce irrelevant content
- Domain relevance checks identify out-of-scope questions
- Empty/malformed input detection prevents nonsensical responses
- Explicit acknowledgment of uncertainty when information is incomplete

## 📊 Evaluation System

The project includes a comprehensive evaluation pipeline for systematically testing and improving the RAG system. This evaluation framework is critical for understanding system performance, identifying weaknesses, and guiding improvements.

### Evaluation Methodology

The evaluation approach follows these principles:

#### 1. 🔍 **Comprehensive Coverage**
The evaluation set includes:
- In-domain questions (derived from FAQ titles)
- Out-of-domain questions (completely unrelated to Voy)
- Edge cases (empty queries, nonsensical input)

#### 2. 📐 **Multi-faceted Assessment**
We evaluate different aspects:
- **Retrieval Quality**: Are the right documents being retrieved?
- **Answer Accuracy**: Does the response match reference information?
- **Self-awareness**: Does the system correctly assess its own confidence?
- **Safety**: Does the system avoid hallucination on out-of-domain questions?

#### 3. 📏 **Quantitative & Qualitative**
Combines:
- Automated metrics (accuracy, response time)
- LLM-based content quality assessment
- Visualizations (confusion matrices)

### Running Evaluations

To run the evaluation pipeline:

```bash
# Create an evaluation dataset
python evaluate.py --create-eval-set --num-samples 50

# Run evaluation on the current model
python evaluate.py --run-eval

# Analyze results and generate metrics
python evaluate.py --analyze

# Evaluate content quality
python evaluate.py --content-eval

# Run all evaluation steps
python evaluate.py --all
```

You can run these steps sequentially or individually. The pipeline automatically saves results at each step and can load them for subsequent steps.

#### Advanced Usage

```bash
# Create a smaller evaluation set for quick testing
python evaluate.py --create-eval-set --num-samples 10

# Specify custom paths for evaluation files
python evaluate.py --eval-set-path ./my_eval_set.json --run-eval

# Analyze results from a specific results file
python evaluate.py --analyze --results-path ./custom_results.json

# Run everything with a custom number of samples
python evaluate.py --all --num-samples 25
```

### Evaluation Metrics

The evaluation system tracks:

#### 1. 🎯 **Confidence Accuracy**
How well the system predicts its own confidence level. This measures the system's self-awareness and ability to recognize its limitations.

#### 2. 🔗 **URL Accuracy**
Whether sources are correctly provided. This evaluates the retrieval component's ability to find relevant documents.

#### 3. ⏱️ **Response Time**
Performance measurements for each query type. This helps identify potential bottlenecks in the system.

#### 4. 🔍 **Content Quality**
LLM-based assessment of answer quality, covering:
- **Relevance**: How well the answer addresses the question
- **Accuracy**: Correctness of information compared to reference
- **Completeness**: Whether all necessary information is provided
- **Clarity**: How clear and understandable the answer is
- **Conciseness**: Whether the answer is appropriately concise
- **Citation**: Proper attribution of information sources

### Evaluation Reports

Evaluation results are saved to the `evaluation/` directory, including:
- `eval_set.json`: Evaluation dataset with expected confidence levels and reference answers
- `results.json`: Raw evaluation results including actual system responses
- `evaluation_report.md`: Summary report with metrics breakdown by category
- `confidence_confusion_matrix.png`: Visualization showing when the system correctly/incorrectly assesses its confidence
- `content_eval_*.json`: Detailed content quality evaluations using LLM assessment

## ❓ Sample Test Questions

1. "What conditions does Voy treat?"
2. "How does the prescription process work?"
3. "What are the side effects of weight loss medication?"

## 🔮 Future Improvements

#### 1. 🔍 **Advanced Retrieval Techniques**
- Implement hybrid search (keyword + semantic)
- Add query expansion for better retrieval
- Explore re-ranking of retrieved documents

#### 2. 📈 **Enhanced Evaluation**
- Integrate human evaluation component
- Add factuality measurement with citation verification
- Implement adversarial testing to find edge cases

#### 3. 🤖 **Model Improvements**
- Experiment with different embedding models
- Explore RAG-fusion for improved answer quality

#### 4. 👥 **User Experience**
- Add conversational context for follow-up questions
- Implement answer generation with step-by-step reasoning




