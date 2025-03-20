import pytest
import sys
import os

# Add the parent directory to the Python path so 'src' can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qa import VoyQASystem

@pytest.fixture
def qa_system():
    return VoyQASystem()

def test_basic_functionality(qa_system):
    """Test that the system returns expected response structure."""
    question = "What is Voy?"
    result = qa_system.answer_question(question)
    
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert "confidence" in result
    assert result["confidence"] in ["high", "low"]

def test_empty_question(qa_system):
    """Test handling of empty questions."""
    question = ""
    result = qa_system.answer_question(question)
    
    assert result["confidence"] == "low"
    assert len(result["sources"]) == 0

def test_medical_disclaimer(qa_system):
    """Test that medical questions include appropriate disclaimers."""
    medical_question = "What are the side effects of weight loss medication?"
    result = qa_system.answer_question(medical_question)
    
    answer_lower = result["answer"].lower()
    assert any(term in answer_lower for term in [
        "consult", "medical", "healthcare", "provider", "doctor", "disclaimer"
    ])

def test_source_citation(qa_system):
    """Test that answers include source citations."""
    question = "How does the prescription process work?"
    result = qa_system.answer_question(question)
    
    assert len(result["sources"]) > 0
    assert all(url.startswith("https://") for url in result["sources"])

def test_hallucination_prevention(qa_system):
    """Test that system handles out-of-scope questions appropriately."""
    irrelevant_question = "What is the capital of France?"
    result = qa_system.answer_question(irrelevant_question)
    
    assert result["confidence"] == "low"
    assert "couldn't find relevant information" in result["answer"] 


if __name__ == "__main__":
    pytest.main()