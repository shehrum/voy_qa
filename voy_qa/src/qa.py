import os
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

load_dotenv()

SYSTEM_TEMPLATE = """You are a helpful assistant for Voy's telehealth weight loss service. Your role is to provide accurate information based on Voy's official documentation.

Important guidelines:
1. Only provide information that is supported by the source documents
2. If you're unsure or the information isn't in the documents, say so
3. Include clear medical disclaimers when discussing treatments or medications
4. Always cite your sources using the provided URLs
5. Do not make up or infer medical information

Context from Voy's documentation:
{context}

Question: {question}

Remember to:
- Be clear and concise
- Include relevant source URLs
- Add medical disclaimers when appropriate
- Express uncertainty when information is incomplete"""

class VoyQASystem:
    def __init__(self, data_path: str = '../data/faq_data.json'):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1
        )
        self.vector_store = None
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
        self.chain = self.prompt | self.llm
        self.load_and_index_data(data_path)

    def load_and_index_data(self, data_path: str):
        """Load FAQ data and create vector store."""
        with open(data_path, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)

        # Create documents from FAQ data
        documents = []
        for item in faq_data:
            text = f"Title: {item['title']}\n\nContent: {item['content']}"
            documents.append(
                Document(
                    page_content=text,
                    metadata={"url": item['url'], "title": item['title']}
                )
            )

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )

    def get_relevant_context(self, question: str, k: int = 3) -> Tuple[str, List[str]]:
        """Retrieve relevant context for the question."""
        results = self.vector_store.similarity_search_with_relevance_scores(question, k=k)
        
        context_texts = []
        urls = []
        
        # Set a minimum relevance threshold - this value might need adjustment
        min_relevance = 0.7
        
        for doc, score in results:
            if score >= min_relevance:  # Only include results with sufficient relevance
                context_texts.append(doc.page_content)
                urls.append(doc.metadata["url"])
        
        return "\n\n".join(context_texts), urls

    def answer_question(self, question: str) -> Dict[str, str]:
        """Answer a question using the Q&A system."""
        # Handle empty questions
        if not question.strip():
            return {
                "answer": "I apologize, but you haven't provided a question. Please ask a specific question about Voy's services.",
                "sources": [],
                "confidence": "low"
            }
        
        context, urls = self.get_relevant_context(question)
        
        # Handle irrelevant or out-of-scope questions
        if not context or not urls:
            return {
                "answer": "I apologize, but I couldn't find relevant information in Voy's documentation to answer your question accurately. Please contact Voy's support team directly for more information.",
                "sources": [],
                "confidence": "low"
            }

        # For questions clearly unrelated to Voy's domain
        if "voy" not in question.lower() and len(urls) < 2:
            # If the question doesn't mention Voy and we have few sources, treat with caution
            return {
                "answer": "I apologize, but I couldn't find relevant information in Voy's documentation to answer your question accurately. Please contact Voy's support team directly for more information.",
                "sources": [],
                "confidence": "low"
            }

        response = self.chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "answer": response.content,
            "sources": urls,
            "confidence": "high"  # Only reach here if we have good context and urls
        }

if __name__ == "__main__":
    # Test the QA system
    qa_system = VoyQASystem()
    test_questions = [
        "What conditions does Voy treat?",
        "How does the prescription process work?",
        "What are the side effects of weight loss medication?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        result = qa_system.answer_question(question)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Confidence: {result['confidence']}") 