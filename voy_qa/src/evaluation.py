import json
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from pathlib import Path
import time

from .qa import VoyQASystem


class RAGEvaluator:
    """Evaluation framework for testing and benchmarking the RAG system."""

    def __init__(self, 
                 qa_system: Optional[VoyQASystem] = None, 
                 data_path: str = '../data/faq_data.json',
                 eval_set_path: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            qa_system: An instance of VoyQASystem or None (will create one)
            data_path: Path to the source data
            eval_set_path: Optional path to evaluation dataset
        """
        self.qa_system = qa_system or VoyQASystem(data_path)
        self.data_path = data_path
        self.eval_set_path = eval_set_path
        self.evaluation_results = []
        
        # Create evaluation directory if it doesn't exist
        Path("../evaluation").mkdir(exist_ok=True)
        Path("./evaluation").mkdir(exist_ok=True)  # Also try current directory
    
    def create_evaluation_set(self, 
                             num_samples: int = 50, 
                             output_path: str = None) -> List[Dict[str, Any]]:
        """
        Create an evaluation dataset from the FAQ data.
        
        Args:
            num_samples: Number of evaluation examples to generate
            output_path: Where to save the evaluation set
            
        Returns:
            List of evaluation examples
        """
        # Default output path
        if output_path is None:
            # Try different paths
            if Path("../evaluation").exists():
                output_path = '../evaluation/eval_set.json'
            else:
                output_path = './evaluation/eval_set.json'
                
        # Load the FAQ data
        with open(self.data_path, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
        
        # Generate evaluation examples
        eval_set = []
        
        # Add some FAQ-based questions (with ground truth)
        for item in faq_data[:min(num_samples//2, len(faq_data))]:
            # Create a question from the FAQ title
            question = item['title']
            if not question.endswith('?'):
                question = f"{question}?"
                
            eval_set.append({
                "question": question,
                "reference_answer": item['content'],
                "reference_urls": [item['url']],
                "expected_confidence": "high",
                "category": "in_domain"
            })
        
        # Add some out-of-domain questions
        out_of_domain = [
            "What is the capital of France?",
            "How do I bake chocolate chip cookies?",
            "What is the meaning of life?",
            "Who won the World Series in 2022?",
            "How do I change a flat tire?",
            "What are the best vacation spots in Europe?",
            "How do I learn to play the piano?",
            "What's the plot of The Great Gatsby?",
            "How do I grow tomatoes?",
            "What's the history of the Roman Empire?"
        ]
        
        for q in out_of_domain[:min(num_samples//4, len(out_of_domain))]:
            eval_set.append({
                "question": q,
                "reference_answer": "I couldn't find relevant information in Voy's documentation to answer your question accurately.",
                "reference_urls": [],
                "expected_confidence": "low",
                "category": "out_of_domain"
            })
        
        # Add some empty or nonsensical questions
        edge_cases = [
            "",
            " ",
            "?",
            "asdfghjkl",
            "...",
            "123456789"
        ]
        
        for q in edge_cases[:min(num_samples//4, len(edge_cases))]:
            eval_set.append({
                "question": q,
                "reference_answer": "I apologize, but you haven't provided a question.",
                "reference_urls": [],
                "expected_confidence": "low",
                "category": "edge_case"
            })
        
        # Save the evaluation set
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_set, f, indent=2)
            
        self.eval_set_path = output_path
        return eval_set
    
    def load_evaluation_set(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load an existing evaluation set."""
        path = path or self.eval_set_path
        if not path:
            raise ValueError("No evaluation set path specified")
            
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def run_evaluation(self, 
                       eval_set: Optional[List[Dict[str, Any]]] = None, 
                       output_path: str = None) -> pd.DataFrame:
        """
        Run evaluation on the QA system.
        
        Args:
            eval_set: List of evaluation examples or None (will load or create one)
            output_path: Where to save results
            
        Returns:
            DataFrame with evaluation results
        """
        # Default output path
        if output_path is None:
            # Try different paths
            if Path("../evaluation").exists():
                output_path = '../evaluation/results.json'
            else:
                output_path = './evaluation/results.json'
                
        # Get or create evaluation set
        if eval_set is None:
            if self.eval_set_path:
                eval_set = self.load_evaluation_set()
            else:
                eval_set = self.create_evaluation_set()
        
        results = []
        
        # Run evaluation
        for example in tqdm(eval_set, desc="Evaluating"):
            start_time = time.time()
            response = self.qa_system.answer_question(example["question"])
            end_time = time.time()
            
            confidence_match = response["confidence"] == example["expected_confidence"]
            urls_match = all(url in response["sources"] for url in example["reference_urls"]) if example["reference_urls"] else len(response["sources"]) == 0
            
            result = {
                "question": example["question"],
                "category": example.get("category", "unknown"),
                "expected_confidence": example["expected_confidence"],
                "actual_confidence": response["confidence"],
                "confidence_match": confidence_match,
                "expected_urls": example["reference_urls"],
                "actual_urls": response["sources"],
                "urls_match": urls_match,
                "reference_answer": example["reference_answer"],
                "actual_answer": response["answer"],
                "response_time": end_time - start_time
            }
            
            results.append(result)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        self.evaluation_results = results
        return pd.DataFrame(results)
    
    def analyze_results(self, results_df: Optional[pd.DataFrame] = None, 
                        output_dir: str = None) -> Dict[str, Any]:
        """
        Analyze evaluation results and generate metrics.
        
        Args:
            results_df: DataFrame with results or None (will use stored results)
            output_dir: Directory to save analysis files
            
        Returns:
            Dictionary with metrics
        """
        # Default output dir
        if output_dir is None:
            # Try different paths
            if Path("../evaluation").exists():
                output_dir = '../evaluation/'
            else:
                output_dir = './evaluation/'
                
        if results_df is None:
            if self.evaluation_results:
                results_df = pd.DataFrame(self.evaluation_results)
            else:
                # Try to load results from file
                results_files = [
                    f"{output_dir}/results.json",
                    "../evaluation/results.json",
                    "./evaluation/results.json"
                ]
                
                for file_path in results_files:
                    if Path(file_path).exists():
                        print(f"Loading results from {file_path}")
                        with open(file_path, 'r') as f:
                            self.evaluation_results = json.load(f)
                        results_df = pd.DataFrame(self.evaluation_results)
                        break
                
                if results_df is None:
                    raise ValueError("No evaluation results available. Run evaluation first with --run-eval")
        
        # Calculate metrics
        metrics = {}
        
        # Overall metrics
        metrics["total_questions"] = len(results_df)
        metrics["confidence_accuracy"] = results_df["confidence_match"].mean()
        metrics["urls_accuracy"] = results_df["urls_match"].mean()
        metrics["avg_response_time"] = results_df["response_time"].mean()
        
        # Metrics by category
        category_metrics = {}
        for category in results_df["category"].unique():
            cat_df = results_df[results_df["category"] == category]
            category_metrics[category] = {
                "count": len(cat_df),
                "confidence_accuracy": cat_df["confidence_match"].mean(),
                "urls_accuracy": cat_df["urls_match"].mean(),
                "avg_response_time": cat_df["response_time"].mean()
            }
        
        metrics["category_metrics"] = category_metrics
        
        # Create confusion matrix for confidence
        y_true = [1 if conf == "high" else 0 for conf in results_df["expected_confidence"]]
        y_pred = [1 if conf == "high" else 0 for conf in results_df["actual_confidence"]]
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["low", "high"])
        disp.plot(ax=ax)
        plt.title("Confidence Level Confusion Matrix")
        plt.savefig(f"{output_dir}/confidence_confusion_matrix.png")
        
        # Generate and save summary report
        report = f"""
        # RAG Evaluation Summary
        
        ## Overall Metrics
        - Total questions: {metrics['total_questions']}
        - Confidence accuracy: {metrics['confidence_accuracy']:.2f}
        - URLs accuracy: {metrics['urls_accuracy']:.2f}
        - Average response time: {metrics['avg_response_time']:.2f} seconds
        
        ## Category Metrics
        """
        
        for category, cat_metrics in category_metrics.items():
            report += f"""
        ### {category.capitalize()}
        - Count: {cat_metrics['count']}
        - Confidence accuracy: {cat_metrics['confidence_accuracy']:.2f}
        - URLs accuracy: {cat_metrics['urls_accuracy']:.2f}
        - Average response time: {cat_metrics['avg_response_time']:.2f} seconds
            """
        
        with open(f"{output_dir}/evaluation_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        return metrics


class ContentEvaluator:
    """Evaluates the quality and correctness of generated content."""
    
    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        """Initialize the content evaluator."""
        from langchain_openai import ChatOpenAI
        
        self.evaluator_llm = ChatOpenAI(
            model=model_name,
            temperature=0
        )
    
    def evaluate_answer_quality(self, 
                               question: str,
                               generated_answer: str,
                               reference_answer: str = None,
                               criteria: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated answer.
        
        Args:
            question: The question that was asked
            generated_answer: The answer generated by the RAG system
            reference_answer: Optional reference answer to compare against
            criteria: List of evaluation criteria or None for defaults
            
        Returns:
            Dictionary with evaluation results
        """
        from langchain.prompts import ChatPromptTemplate
        
        if criteria is None:
            criteria = [
                "Relevance - How well does the answer address the question?",
                "Accuracy - Is the information correct based on the reference?",
                "Completeness - Does the answer provide all necessary information?",
                "Clarity - Is the answer clear and easy to understand?",
                "Conciseness - Is the answer appropriately concise?",
                "Citation - Are sources properly cited?"
            ]
        
        criteria_text = "\n".join([f"- {c}" for c in criteria])
        
        reference_context = f"\nReference answer: {reference_answer}" if reference_answer else ""
        
        prompt = ChatPromptTemplate.from_template(
            """You are an expert evaluator of question answering systems.
            
            Please evaluate the following answer to a given question based on these criteria:
            {criteria}
            
            Question: {question}
            Generated answer: {answer}
            Reference answer: {reference}
            
            For each criterion, provide a score from 1-5 (where 5 is best) and a brief explanation.
            Then provide an overall score from a 1-5 scale and a summary of the evaluation.
            
            Format your response as JSON with the following structure:
            {{
                "criteria": {{
                    "criterion_name": {{
                        "score": score_value,
                        "explanation": "explanation_text"
                    }},
                    ...
                }},
                "overall": {{
                    "score": overall_score_value,
                    "explanation": "overall_explanation"
                }}
            }}
            """
        )
        
        evaluation_result = self.evaluator_llm.invoke(
            prompt.format(
                criteria=criteria_text,
                question=question,
                answer=generated_answer,
                reference=reference_context
            )
        )
        
        # Parse the result as JSON
        import json
        try:
            content = evaluation_result.content
            # Clean up potential markdown JSON formatting
            if content.startswith('```json'):
                content = content.strip('```json').strip('```').strip()
            result_json = json.loads(content)
            return result_json
        except json.JSONDecodeError:
            # If parsing fails, return the raw result
            return {
                "error": "Failed to parse evaluation result as JSON",
                "raw_result": evaluation_result.content
            }


def main():
    """Run evaluation as a script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--create-eval-set", action="store_true", help="Create a new evaluation set")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples for evaluation set")
    parser.add_argument("--run-eval", action="store_true", help="Run evaluation")
    parser.add_argument("--analyze", action="store_true", help="Analyze results")
    parser.add_argument("--content-eval", action="store_true", help="Evaluate content quality")
    parser.add_argument("--eval-set-path", type=str, help="Path to evaluation set")
    parser.add_argument("--results-path", type=str, help="Path to results file")
    parser.add_argument("--all", action="store_true", help="Run all steps: create, run, analyze, and content evaluation")
    
    args = parser.parse_args()
    
    # If --all is specified, set all individual flags
    if args.all:
        args.create_eval_set = True
        args.run_eval = True
        args.analyze = True
        args.content_eval = True
    
    # Create QA system
    qa_system = VoyQASystem()
    evaluator = RAGEvaluator(qa_system, eval_set_path=args.eval_set_path)
    
    if args.create_eval_set:
        print(f"Creating evaluation set with {args.num_samples} samples...")
        evaluator.create_evaluation_set(num_samples=args.num_samples)
        print(f"Evaluation set created at {evaluator.eval_set_path}")
    
    if args.run_eval:
        print("Running evaluation...")
        results_df = evaluator.run_evaluation()
        print(f"Evaluation completed. Results saved.")
    
    if args.analyze:
        print("Analyzing results...")
        metrics = evaluator.analyze_results()
        print(f"Analysis completed. Report saved in evaluation directory.")
        
        # Display summary table
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total questions: {metrics['total_questions']}")
        print(f"Confidence accuracy: {metrics['confidence_accuracy']:.2f}")
        print(f"URLs accuracy: {metrics['urls_accuracy']:.2f}")
        print(f"Average response time: {metrics['avg_response_time']:.2f} seconds")
        
        print("\nCategory breakdown:")
        for category, cat_metrics in metrics.get("category_metrics", {}).items():
            print(f"  - {category.capitalize()}: {cat_metrics['confidence_accuracy']:.2f} accuracy ({cat_metrics['count']} questions)")
        
        print("\nDetailed report saved to evaluation/evaluation_report.md")
        print("Confusion matrix saved to evaluation/confidence_confusion_matrix.png")
        print("="*50)
    
    if args.content_eval:
        print("Evaluating content quality...")
        content_evaluator = ContentEvaluator()
        
        # Load results if available
        if args.results_path:
            with open(args.results_path, 'r') as f:
                results = json.load(f)
        elif evaluator.evaluation_results:
            results = evaluator.evaluation_results
        else:
            # Try to load results from file
            results_files = [
                "../evaluation/results.json",
                "./evaluation/results.json"
            ]
            
            for file_path in results_files:
                if Path(file_path).exists():
                    print(f"Loading results from {file_path}")
                    with open(file_path, 'r') as f:
                        results = json.load(f)
                    break
            
        if not results:
            print("No results available for content evaluation. Run evaluation first with --run-eval")
            return
        
        # Determine output directory
        if Path("../evaluation").exists():
            output_dir = "../evaluation"
        else:
            output_dir = "./evaluation"
        
        # Evaluate a sample of results
        sample_size = min(5, len(results))
        
        print("\n" + "="*50)
        print("CONTENT EVALUATION RESULTS")
        print("="*50)
        
        all_scores = []
        
        for i in range(sample_size):
            result = results[i]
            question = result['question']
            print(f"\nEvaluating: {question[:50]}{'...' if len(question) > 50 else ''}")
            
            content_eval = content_evaluator.evaluate_answer_quality(
                question=result['question'],
                generated_answer=result['actual_answer'],
                reference_answer=result['reference_answer']
            )
            content_eval['question'] = question
            content_eval['generated_answer'] = result['actual_answer']
            content_eval['reference_answer'] = result['reference_answer']
            # Save detailed evaluation
            with open(f"{output_dir}/content_eval_{i}.json", 'w') as f:
                json.dump(content_eval, f, indent=2)
            
            # Get overall score
            overall_score = content_eval.get('overall', {}).get('score', 'N/A')
            if isinstance(overall_score, (int, float)):
                all_scores.append(overall_score)
            
            # Print result
            print(f"  Score: {overall_score}/5")
            
            # Print individual criteria scores if available
            if 'criteria' in content_eval:
                for criterion, details in content_eval['criteria'].items():
                    score = details.get('score', 'N/A')
                    print(f"  - {criterion}: {score}/5")
        
        # Print average score
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            print(f"\nAverage content quality score: {avg_score:.2f}/5")
        
        print(f"\nDetailed evaluations saved to {output_dir}/content_eval_*.json")
        print("="*50)


if __name__ == "__main__":
    main() 