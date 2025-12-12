"""
deepeval wrapper service for optional DeepEval metrics.
This file imports external `deepeval` package lazily and raises a clear error
if it's not installed so the API endpoint can return a helpful message.
"""
from src.search import RAGSearch
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from deepeval.test_case import LLMTestCase  # type: ignore
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric  # type: ignore


class DeepEvalService:
    def __init__(self, rag_search: RAGSearch):
        self.rag_search = rag_search
        try:
            # Lazy import external dependency and bind to instance
            from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric  # type: ignore
            from deepeval.test_case import LLMTestCase  # type: ignore
        except ImportError as e:
            raise ImportError(
                "DeepEval package not installed. Install it with: pip install deepeval"
            ) from e

        self.AnswerRelevancyMetric = AnswerRelevancyMetric
        self.FaithfulnessMetric = FaithfulnessMetric
        self.ContextualRelevancyMetric = ContextualRelevancyMetric
        self.LLMTestCase = LLMTestCase

    def evaluate_query(self, query: str, expected_answer: str = None):
        """Evaluate a query dynamically using the live RAG KB"""

        # Step 1: Retrieve results and summary
        raw_results = self.rag_search.search_only(query, top_k=15)
        answer = self.rag_search.search_and_summarize(query, top_k=15)
        retrieval_context = [r["metadata"]["text"] for r in raw_results]

        # Step 2: Create test case
        test_case = self.LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=retrieval_context,
            expected_output=expected_answer
        )

        results = {}

        # Step 3: Run metrics
        try:
            metric = self.AnswerRelevancyMetric(threshold=0.7)
            metric.measure(test_case)
            results["answer_relevancy"] = {"score": metric.score, "passed": metric.is_successful()}
        except Exception as e:
            results["answer_relevancy"] = {"error": str(e)}

        try:
            metric = self.FaithfulnessMetric(threshold=0.7)
            metric.measure(test_case)
            results["faithfulness"] = {"score": metric.score, "passed": metric.is_successful()}
        except Exception as e:
            results["faithfulness"] = {"error": str(e)}

        try:
            metric = self.ContextualRelevancyMetric(threshold=0.7)
            metric.measure(test_case)
            results["contextual_relevancy"] = {"score": metric.score, "passed": metric.is_successful()}
        except Exception as e:
            results["contextual_relevancy"] = {"error": str(e)}

        return {
            "query": query,
            "answer": answer,
            "num_retrieved_chunks": len(retrieval_context),
            "metrics": results
        }
