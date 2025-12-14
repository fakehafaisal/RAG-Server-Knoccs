"""
DeepEval wrapper service for optional DeepEval metrics.
This file imports external `deepeval` package lazily and raises a clear error
if it's not installed so the API endpoint can return a helpful message.
"""
import os
from dotenv import load_dotenv
from src.search import RAGSearch
from typing import Any, TYPE_CHECKING, Dict

load_dotenv()

if TYPE_CHECKING:
    from deepeval.test_case import LLMTestCase  # type: ignore
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric  # type: ignore


class DeepEvalService:
    def __init__(self, rag_search: RAGSearch):
        self.rag_search = rag_search
        
        # Check for OpenAI API key
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "DeepEval requires OpenAI API to evaluate answers. "
                "Add OPENAI_API_KEY=sk-... to your .env file."
            )
        
        try:
            # Lazy import external dependency and bind to instance
            from deepeval.metrics import (
                AnswerRelevancyMetric, 
                FaithfulnessMetric, 
                ContextualRelevancyMetric
            )
            from deepeval.test_case import LLMTestCase
            
            # Set API key for deepeval
            os.environ["OPENAI_API_KEY"] = self.openai_key
            
        except ImportError as e:
            raise ImportError(
                "DeepEval package not installed. Install it with: pip install deepeval"
            ) from e

        self.AnswerRelevancyMetric = AnswerRelevancyMetric
        self.FaithfulnessMetric = FaithfulnessMetric
        self.ContextualRelevancyMetric = ContextualRelevancyMetric
        self.LLMTestCase = LLMTestCase

    def evaluate_query(self, query: str, expected_answer: str = None) -> Dict:
        """Evaluate a query dynamically using the live RAG KB"""

        # Step 1: Retrieve results and summary
        raw_results = self.rag_search.search_only(query, top_k=15)
        answer = self.rag_search.search_and_summarize(query, top_k=15)
        retrieval_context = [r["metadata"]["text"] for r in raw_results]

        # Validate we got results
        if not retrieval_context:
            return {
                "query": query,
                "answer": answer,
                "num_retrieved_chunks": 0,
                "metrics": {
                    "error": "No chunks retrieved - cannot evaluate"
                }
            }

        # Step 2: Create test case
        test_case = self.LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=retrieval_context,
            expected_output=expected_answer
        )

        results = {}

        # Step 3: Run metrics with better error handling
        print(f"[INFO] Running DeepEval metrics for query: {query[:50]}...")
        
        # Answer Relevancy
        try:
            print("[INFO] Evaluating Answer Relevancy...")
            metric = self.AnswerRelevancyMetric(
                threshold=0.7,
                model="gpt-4o-mini",  # Use cheaper model for evaluation
                include_reason=True
            )
            metric.measure(test_case)
            results["answer_relevancy"] = {
                "score": float(metric.score) if metric.score is not None else None,
                "passed": metric.is_successful(),
                "reason": metric.reason if hasattr(metric, 'reason') else None
            }
            print(f"[SUCCESS] Answer Relevancy: {metric.score:.2f}")
        except Exception as e:
            print(f"[ERROR] Answer Relevancy failed: {str(e)}")
            results["answer_relevancy"] = {"error": str(e), "score": None, "passed": False}

        # Faithfulness
        try:
            print("[INFO] Evaluating Faithfulness...")
            metric = self.FaithfulnessMetric(
                threshold=0.7,
                model="gpt-4o-mini",
                include_reason=True
            )
            metric.measure(test_case)
            results["faithfulness"] = {
                "score": float(metric.score) if metric.score is not None else None,
                "passed": metric.is_successful(),
                "reason": metric.reason if hasattr(metric, 'reason') else None
            }
            print(f"[SUCCESS] Faithfulness: {metric.score:.2f}")
        except Exception as e:
            print(f"[ERROR] Faithfulness failed: {str(e)}")
            results["faithfulness"] = {"error": str(e), "score": None, "passed": False}

        # Contextual Relevancy
        try:
            print("[INFO] Evaluating Contextual Relevancy...")
            metric = self.ContextualRelevancyMetric(
                threshold=0.7,
                model="gpt-4o-mini",
                include_reason=True
            )
            metric.measure(test_case)
            results["contextual_relevancy"] = {
                "score": float(metric.score) if metric.score is not None else None,
                "passed": metric.is_successful(),
                "reason": metric.reason if hasattr(metric, 'reason') else None
            }
            print(f"[SUCCESS] Contextual Relevancy: {metric.score:.2f}")
        except Exception as e:
            print(f"[ERROR] Contextual Relevancy failed: {str(e)}")
            results["contextual_relevancy"] = {"error": str(e), "score": None, "passed": False}

        return {
            "query": query,
            "answer": answer,
            "num_retrieved_chunks": len(retrieval_context),
            "metrics": results
        }

