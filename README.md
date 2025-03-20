# Voy Q&A System

This project implements a Q&A system for Voy's telehealth services using LLMs and document retrieval. The system leverages retrieval-augmented generation (RAG) to provide accurate, relevant, and safe answers to user questions about Voy's telehealth weight loss services.

## Project Structure

```
voy_qa/
├── src/             # Source code
│   ├── scraper.py   # FAQ scraping functionality
│   ├── qa.py        # Q&A system implementation
│   ├── evaluation.py # Evaluation framework
│   └── app.py       # Gradio web interface
├── tests/           # Test cases
└── evaluate.py      # Evaluation script
├── data/            # Stores scraped FAQ data and embeddings
├── evaluation/      # Stores evaluation results and reports

```

## Setup

1. Create a virtual environment:
```bash
conda create -n voy_qa python=3.9
conda activate voy_qa
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_key_here
```

4. Usage

Run the Scraper:
```bash
python src/scraper.py
```


Run the Gradio interface:
```bash
python src/app.py
```

Run tests:
```bash
pytest tests/
```

## Architecture & Design Choices

### RAG System Architecture

The VoyQA system follows a modern retrieval-augmented generation architecture with the following components:

1. **Document Processing Pipeline**
   - Data is sourced from Voy's FAQ pages and structured as JSON
   - Documents are split into optimal chunks (1000 tokens with 200 token overlap)
   - Document metadata preserves original source URLs for attribution

2. **Vector Database & Retrieval**
   - Uses Chroma as the vector database for efficient similarity search
   - OpenAI embeddings create dense vector representations
   - Implements relevance threshold filtering (0.7) to prevent low-quality retrievals
   - Returns multiple sources (k=3) to provide comprehensive context

3. **LLM Integration**
   - Uses GPT-4 Turbo with low temperature (0.1) for consistent, factual responses
   - Custom prompt template enforces accuracy, citation, and medical disclaimers
   - Chain-based architecture enables future extension to more complex reasoning flows

4. **Confidence Assessment**
   - Multi-level confidence scoring based on:
     - Presence of relevant retrieved documents
     - Question relevance to Voy's domain
     - Explicit checks for empty or malformed queries
   - Low confidence triggers safe fallback responses

### System Design 

The system was designed with several key principles in mind:

1. **Accuracy First**: The system prioritizes factual accuracy by:
   - Only using information from vetted source documents
   - Setting strict relevance thresholds for retrieval
   - Including source attribution in all responses
   - Clearly expressing uncertainty when information is limited

2. **Medical Safety**: Given the health-related context, special consideration is given to:
   - Including medical disclaimers for treatment-related questions
   - Avoiding inference or extrapolation beyond source material
   - Directing users to healthcare professionals for medical advice
   - Providing clear confidence indicators for all responses

3. **Extensibility**: The architecture allows for:
   - Easy addition of new data sources
   - Swapping of embedding or LLM models
   - Integration of more sophisticated retrieval techniques
   - Enhanced evaluation methodologies

## Features

- Web scraping of Voy FAQ pages
- Document retrieval using ChromaDB
- LLM-powered question answering with OpenAI
- Gradio web interface
- Evaluation metrics and safety measures

## Safety Measures

1. **Source Attribution**: All responses include references to source documents, allowing users to verify information.

2. **Confidence Scoring**: The system assesses its confidence in each response:
   - **High Confidence**: Question is relevant to Voy's domain, multiple high-quality sources retrieved
   - **Low Confidence**: Question is out-of-domain, no relevant sources found, or input is malformed

3. **Medical Disclaimer**: Clear disclaimers are included when discussing treatments or medications, emphasizing that the system does not provide medical advice.

4. **Hallucination Prevention**: 
   - Strict retrieval thresholds reduce irrelevant content
   - Domain relevance checks identify out-of-scope questions
   - Empty/malformed input detection prevents nonsensical responses
   - Explicit acknowledgment of uncertainty when information is incomplete

## Evaluation System

The project includes a comprehensive evaluation pipeline for systematically testing and improving the RAG system. This evaluation framework is critical for understanding system performance, identifying weaknesses, and guiding improvements.

### Evaluation Methodology

The evaluation approach follows these principles:

1. **Comprehensive Coverage**: The evaluation set includes:
   - In-domain questions (derived from FAQ titles)
   - Out-of-domain questions (completely unrelated to Voy)
   - Edge cases (empty queries, nonsensical input)

2. **Multi-faceted Assessment**: We evaluate different aspects:
   - **Retrieval Quality**: Are the right documents being retrieved?
   - **Answer Accuracy**: Does the response match reference information?
   - **Self-awareness**: Does the system correctly assess its own confidence?
   - **Safety**: Does the system avoid hallucination on out-of-domain questions?

3. **Quantitative & Qualitative**: Combines:
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

1. **Confidence Accuracy**: How well the system predicts its own confidence level. This measures the system's self-awareness and ability to recognize its limitations.

2. **URL Accuracy**: Whether sources are correctly provided. This evaluates the retrieval component's ability to find relevant documents.

3. **Response Time**: Performance measurements for each query type. This helps identify potential bottlenecks in the system.

4. **Content Quality**: LLM-based assessment of answer quality, covering:
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

## Sample Test Questions

1. "What conditions does Voy treat?"
2. "How does the prescription process work?"
3. "What are the side effects of weight loss medication?"


## Future Improvements


1. **Advanced Retrieval Techniques**:
   - Implement hybrid search (keyword + semantic)
   - Add query expansion for better retrieval
   - Explore re-ranking of retrieved documents

2. **Enhanced Evaluation**:
   - Integrate human evaluation component
   - Add factuality measurement with citation verification
   - Implement adversarial testing to find edge cases

3. **Model Improvements**:
   - Experiment with different embedding models
   - Explore RAG-fusion for improved answer quality

4. **User Experience**:
   - Add conversational context for follow-up questions
   - Implement answer generation with step-by-step reasoning


