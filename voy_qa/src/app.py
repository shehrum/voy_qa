import os
import gradio as gr
from qa import VoyQASystem
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize the QA system with error handling
try:
    qa_system = VoyQASystem()
    qa_system_initialized = True
except Exception as e:
    logger.error(f"Failed to initialize QA system: {str(e)}")
    qa_system_initialized = False

def format_response(result: dict) -> str:
    """Format the QA system response for display."""
    answer = result["answer"]
    sources = result["sources"]
    confidence = result["confidence"]
    
    response = f"{answer}\n\n"
    
    if sources:
        response += "\nSources:\n"
        for url in sources:
            response += f"- {url}\n"
    
    response += f"\nConfidence: {confidence}"
    
    if confidence == "low":
        response += "\n⚠️ Note: This response has low confidence. Please verify with Voy's support team."
    
    return response

def answer_question(question: str) -> str:
    """Process the question and return a formatted response."""
    if not question.strip():
        return "Please enter a question."
    
    if not qa_system_initialized:
        return "The QA system is currently unavailable. Please check your API keys and quotas."
    
    try:
        result = qa_system.answer_question(question)
        return format_response(result)
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return f"An error occurred: {str(e)}"

# Create the Gradio interface
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask a question about Voy's telehealth services..."
    ),
    outputs=gr.Textbox(
        label="Answer",
        lines=10
    ),
    title="Voy Telehealth Q&A Assistant",
    description="""
    Ask questions about Voy's telehealth services and weight loss treatments.
    
    **Medical Disclaimer**: This is an AI assistant that provides information based on Voy's FAQ documentation.
    It is not a substitute for professional medical advice. Always consult with healthcare providers for
    medical decisions.
    """,
    examples=[
        ["What conditions does Voy treat?"],
        ["How does the prescription process work?"],
        ["What are the side effects of weight loss medication?"],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=True,
            debug=False
        )
    except Exception as e:
        logger.error(f"Failed to launch interface: {str(e)}")
        print(f"Error: {str(e)}") 