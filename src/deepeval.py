
import os
import json
from dotenv import load_dotenv
from src.search import RAGSearch
from typing import Dict, Optional

load_dotenv()


class DeepEvalService:
    def __init__(self, rag_search: RAGSearch, ground_truth_file: str = "ground_truth.json"):
        """
        Initialize DeepEval with OpenAI for KNOCCS KB evaluation
        
        Args:
            rag_search: Your RAG search instance
            ground_truth_file: (Deprecated - kept for compatibility, not used)
        """
        self.rag_search = rag_search
        
        # Check for OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Add OPENAI_API_KEY=sk-... to your .env file."
            )
        
        try:
            # Import metrics
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                FaithfulnessMetric,
                ContextualPrecisionMetric
            )
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams
            from deepeval.models import GPTModel
            
            # Initialize OpenAI model (built-in support)
            self.openai_model = GPTModel(model="gpt-5")
            
            self.AnswerRelevancyMetric = AnswerRelevancyMetric
            self.FaithfulnessMetric = FaithfulnessMetric
            self.ContextualPrecisionMetric = ContextualPrecisionMetric
            self.LLMTestCase = LLMTestCase
            self.LLMTestCaseParams = LLMTestCaseParams
            
            print("DeepEval initialized with OpenAI GPT-5 + 3 standard metrics for KNOCCS")
            print("Running in referenceless mode (no ground truth required)")
            
        except ImportError as e:
            raise ImportError(
                "DeepEval package not installed. Install it with: pip install deepeval"
            ) from e

    def evaluate_query(self, query: str) -> Dict:
        """
        Evaluate a query using 3 standard RAG metrics (REFERENCELESS)
        No ground truth needed - purely evaluates answer quality and retrieval
        
        Args:
            query: The user query to evaluate
        
        Returns:
            Dictionary with evaluation results from all 3 metrics
        """
        # Step 1: Retrieve results and generate answer
        try:
            raw_results = self.rag_search.search_only(query, top_k=15)
            answer = self.rag_search.search_and_summarize(query, top_k=15)
            retrieval_context = [r["metadata"]["text"] for r in raw_results]
        except Exception as e:
            print(f"[ERROR] Failed to retrieve results: {e}")
            return {
                "query": query,
                "answer": f"Error: {str(e)}",
                "expected_answer": None,
                "num_retrieved_chunks": 0,
                "metrics": {"error": f"Retrieval failed: {str(e)}"}
            }

        # Validate we got results
        if not retrieval_context:
            return {
                "query": query,
                "answer": answer,
                "expected_answer": None,
                "num_retrieved_chunks": 0,
                "metrics": {
                    "error": "No chunks retrieved - cannot evaluate"
                }
            }

        print(f"Retrieved {len(retrieval_context)} chunks")
        print(f"Answer length: {len(answer)} characters")

        # Step 2: Create test case (NO expected_output needed for referenceless)
        test_case = self.LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=retrieval_context
        )

        results = {}

        # ============ METRIC 1: Answer Relevancy ============
        try:
            print("[INFO] → Evaluating Answer Relevancy...")
            metric = self.AnswerRelevancyMetric(model=self.openai_model, threshold=0.7)
            metric.measure(test_case)
            
            score = float(metric.score) if metric.score is not None else 0.0
            results["answer_relevancy"] = {
                "score": score,
                "passed": metric.is_successful(),
                "reason": getattr(metric, 'reason', None),
                "description": "Does the answer actually address the user's question?"
            }
            status = "✓ PASS" if metric.is_successful() else "✗ FAIL"
            print(f"[SUCCESS] Answer Relevancy: {score:.4f} {status}")
            
        except Exception as e:
            print(f"[ERROR] Answer Relevancy failed: {str(e)}")
            results["answer_relevancy"] = {
                "error": str(e), 
                "score": 0.0, 
                "passed": False,
                "description": "Does the answer actually address the user's question?"
            }

        # ============ METRIC 2: Faithfulness ============
        try:
            print("[INFO] → Evaluating Faithfulness (Hallucination Check)...")
            metric = self.FaithfulnessMetric(model=self.openai_model, threshold=0.7, async_mode=False)
            metric.measure(test_case)
            
            score = float(metric.score) if metric.score is not None else 0.0
            results["faithfulness"] = {
                "score": score,
                "passed": metric.is_successful(),
                "reason": getattr(metric, 'reason', None),
                "description": "Is the answer factually consistent with the retrieved context?"
            }
            status = "✓ PASS" if metric.is_successful() else "✗ FAIL"
            print(f"[SUCCESS] Faithfulness: {score:.4f} {status}")
            
        except Exception as e:
            print(f"[ERROR] Faithfulness failed: {str(e)}")
            results["faithfulness"] = {
                "error": str(e), 
                "score": 0.0, 
                "passed": False,
                "description": "Is the answer factually consistent with the retrieved context?"
            }

        # ============ METRIC 3: Contextual Precision ============
        try:
            print("[INFO] → Evaluating Contextual Precision (Retriever Quality)...")
            metric = self.ContextualPrecisionMetric(model=self.openai_model, threshold=0.7)
            metric.measure(test_case)
            
            score = float(metric.score) if metric.score is not None else 0.0
            results["contextual_precision"] = {
                "score": score,
                "passed": metric.is_successful(),
                "reason": getattr(metric, 'reason', None),
                "description": "Are the retrieved chunks focused and relevant without noise?"
            }
            status = "✓ PASS" if metric.is_successful() else "✗ FAIL"
            print(f"[SUCCESS] Contextual Precision: {score:.4f} {status}")
            
        except Exception as e:
            print(f"[ERROR] Contextual Precision failed: {str(e)}")
            results["contextual_precision"] = {
                "error": str(e), 
                "score": 0.0, 
                "passed": False,
                "description": "Are the retrieved chunks focused and relevant without noise?"
            }

        print(f"[INFO] Evaluation complete!\n")
        
        return {
            "query": query,
            "answer": answer,
            "expected_answer": None,  # Always None in referenceless mode
            "num_retrieved_chunks": len(retrieval_context),
            "metrics": results
        }


# At the bottom of deepeval.py

if __name__ == "__main__":
    from src.search import RAGSearch
    
    # Initialize RAG
    rag = RAGSearch(debug=False)
    
    # Initialize DeepEval (referenceless mode)
    print("\n[INFO] Initializing DeepEval Service...")
    evaluator = DeepEvalService(rag)
    
    # Test single query
    print("\n[INFO] Running single query evaluation...")
    result = evaluator.evaluate_query("What is Pulse Unity's main business focus?")
    print(f"\nResult: {result}")