import os
import json
from dotenv import load_dotenv
from typing import Dict, Optional, List
from difflib import SequenceMatcher

# Handle imports properly
try:
    from src.search import RAGSearch
    from src.config import LLMConfig, DeepEvalConfig, SearchConfig
    from src.logger import get_deepeval_logger
except ModuleNotFoundError:
    from search import RAGSearch
    from config import LLMConfig, DeepEvalConfig, SearchConfig
    from logger import get_deepeval_logger

load_dotenv()

# Initialize logger
logger = get_deepeval_logger()

# Configure DeepEval timeout settings (set before importing deepeval)
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", str(DeepEvalConfig.PER_TASK_TIMEOUT))
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", str(DeepEvalConfig.PER_ATTEMPT_TIMEOUT))
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", str(DeepEvalConfig.RETRY_MAX_ATTEMPTS))


class DeepEvalService:
    def __init__(self, rag_search: RAGSearch, ground_truth_file: str = "ground_truth.json"):
        """
        Initialize DeepEval with OpenAI for KNOCCS KB evaluation

        Args:
            rag_search: Your RAG search instance
            ground_truth_file: Path to ground truth JSON file for reference-based evaluation
        """
        self.rag_search = rag_search
        self.ground_truth_data = []
        self.ground_truth_file = ground_truth_file

        # Load ground truth file
        self._load_ground_truth(ground_truth_file)

        # Check for OpenAI API key
        openai_key = LLMConfig.OPENAI_API_KEY
        if not openai_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Add OPENAI_API_KEY=sk-... to your .env file."
            )

        try:
            # Check DeepEval version
            import deepeval
            deepeval_version = getattr(deepeval, '__version__', 'unknown')
            logger.info(f"DeepEval version: {deepeval_version}")

            # Import metrics and evaluate function
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                FaithfulnessMetric,
                ContextualPrecisionMetric
            )
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams
            from deepeval.models import GPTModel
            from deepeval import evaluate

            # Try to import AsyncConfig (may not exist in older versions)
            try:
                from deepeval.evaluate import AsyncConfig
                self.AsyncConfig = AsyncConfig
                self.has_async_config = True
                logger.info("AsyncConfig available - parallel evaluation enabled")
            except ImportError:
                self.AsyncConfig = None
                self.has_async_config = False
                logger.warning("AsyncConfig not available - will use sequential evaluation")

            # Initialize OpenAI model
            self.openai_model = GPTModel(model=LLMConfig.DEEPEVAL_MODEL)

            self.AnswerRelevancyMetric = AnswerRelevancyMetric
            self.FaithfulnessMetric = FaithfulnessMetric
            self.ContextualPrecisionMetric = ContextualPrecisionMetric
            self.LLMTestCase = LLMTestCase
            self.LLMTestCaseParams = LLMTestCaseParams
            self.evaluate = evaluate

            logger.info(f"DeepEval initialized with OpenAI {LLMConfig.DEEPEVAL_MODEL}")
            logger.info(f"Running in reference-based mode with {len(self.ground_truth_data)} ground truth entries")
            logger.info(f"Timeout settings: {DeepEvalConfig.PER_TASK_TIMEOUT}s per task")

        except ImportError as e:
            raise ImportError(
                f"DeepEval package not installed or import failed. Install it with: pip install deepeval. Error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize DeepEval: {e}"
            ) from e

    def _load_ground_truth(self, ground_truth_file: str):
        """Load ground truth data from JSON file"""
        try:
            if os.path.exists(ground_truth_file):
                with open(ground_truth_file, 'r') as f:
                    self.ground_truth_data = json.load(f)
                logger.info(f"Loaded {len(self.ground_truth_data)} ground truth entries from {ground_truth_file}")
            else:
                logger.warning(f"Ground truth file not found: {ground_truth_file}")
                self.ground_truth_data = []
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            self.ground_truth_data = []

    def _find_matching_ground_truth(self, query: str) -> Optional[Dict]:
        """
        Find matching ground truth entry for a given query using fuzzy matching
        Returns the ground truth entry if found, None otherwise
        """
        if not self.ground_truth_data:
            return None

        best_match = None
        best_ratio = 0.0
        threshold = SearchConfig.GROUND_TRUTH_THRESHOLD

        for entry in self.ground_truth_data:
            gt_query = entry.get("query", "").lower()
            query_lower = query.lower()

            # Calculate similarity ratio
            ratio = SequenceMatcher(None, query_lower, gt_query).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = entry

        if best_ratio >= threshold:
            logger.info(f"Found matching ground truth (similarity: {best_ratio:.2%})")
            return best_match

        logger.info(f"No matching ground truth found (best match: {best_ratio:.2%})")
        return None

    def evaluate_query(self, query: str) -> Dict:
        """
        Evaluate a query using DeepEval metrics with parallel execution.

        Uses evaluate() with run_async=True for faster parallel metric evaluation
        instead of sequential metric.measure() calls.
        """
        # Step 1: Retrieve results and generate answer
        try:
            raw_results = self.rag_search.search_only(query, top_k=SearchConfig.TOP_K)
            answer, _ = self.rag_search.search_and_summarize(
                query, top_k=SearchConfig.TOP_K, conversation_history=[]
            )
            retrieval_context = [r["metadata"]["text"] for r in raw_results]
        except Exception as e:
            logger.error(f"Failed to retrieve results: {e}")
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

        logger.info(f"Retrieved {len(retrieval_context)} chunks")
        logger.info(f"Answer length: {len(answer)} characters")

        # Step 2: Find matching ground truth
        ground_truth_match = self._find_matching_ground_truth(query)
        expected_answer = None
        if ground_truth_match:
            expected_answer = ground_truth_match.get("expected_answer")
            logger.info("Using expected answer from ground truth")

        # Step 3: Create test case with ground truth (if available)
        test_case = self.LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=retrieval_context,
            expected_output=expected_answer
        )

        # Step 4: Create all metrics upfront
        metrics = [
            self.AnswerRelevancyMetric(
                model=self.openai_model,
                threshold=DeepEvalConfig.METRIC_THRESHOLD
            ),
            self.FaithfulnessMetric(
                model=self.openai_model,
                threshold=DeepEvalConfig.METRIC_THRESHOLD
            ),
            self.ContextualPrecisionMetric(
                model=self.openai_model,
                threshold=DeepEvalConfig.METRIC_THRESHOLD
            )
        ]

        metric_descriptions = {
            "answer_relevancy": "Does the answer actually address the user's question?",
            "faithfulness": "Is the answer factually consistent with the retrieved context?",
            "contextual_precision": "Are the retrieved chunks focused and relevant without noise?"
        }

        # Step 5: Run all metrics using evaluate()
        results = {}

        try:
            # Build evaluate() kwargs based on available features
            eval_kwargs = {
                "test_cases": [test_case],
                "metrics": metrics,
            }

            # Add async config if available (for parallel execution)
            if self.has_async_config and self.AsyncConfig:
                logger.info("Running all 3 metrics in parallel (async mode)...")
                try:
                    async_config = self.AsyncConfig(
                        run_async=True,
                        max_concurrent=DeepEvalConfig.MAX_CONCURRENT
                    )
                    eval_kwargs["async_config"] = async_config
                except Exception as async_err:
                    logger.warning(f"Failed to create AsyncConfig: {async_err}")
                    logger.info("Falling back to sequential evaluation...")
            else:
                logger.info("Running metrics sequentially...")

            # Use DeepEval's evaluate()
            eval_result = self.evaluate(**eval_kwargs)

            # Process results from eval_result (DeepEval 3.8.x returns results in test_results)
            # Map metric names to our internal names (handles multiple formats)
            metric_name_map = {
                # Class names
                "AnswerRelevancyMetric": "answer_relevancy",
                "FaithfulnessMetric": "faithfulness",
                "ContextualPrecisionMetric": "contextual_precision",
                # Display names (from DeepEval output)
                "Answer Relevancy": "answer_relevancy",
                "Faithfulness": "faithfulness",
                "Contextual Precision": "contextual_precision",
            }

            # Extract results from eval_result.test_results
            if hasattr(eval_result, 'test_results') and eval_result.test_results:
                test_result = eval_result.test_results[0]
                if hasattr(test_result, 'metrics_data'):
                    for metric_data in test_result.metrics_data:
                        # Get metric name from the metric data
                        raw_name = getattr(metric_data, 'name', '')
                        metric_name = metric_name_map.get(raw_name)
                        # Fallback: try lowercase with underscores
                        if not metric_name:
                            metric_name = raw_name.lower().replace(' ', '_').replace('metric', '').strip('_')

                        if metric_name in metric_descriptions:
                            score = float(metric_data.score) if metric_data.score is not None else 0.0
                            passed = metric_data.success if hasattr(metric_data, 'success') else score >= DeepEvalConfig.METRIC_THRESHOLD
                            reason = getattr(metric_data, 'reason', None)

                            results[metric_name] = {
                                "score": score,
                                "passed": passed,
                                "reason": reason,
                                "description": metric_descriptions[metric_name]
                            }

                            status = "PASS" if passed else "FAIL"
                            logger.info(f"{metric_name}: {score:.4f} {status}")

            # Fallback: if no results extracted from eval_result, try metric objects
            if not results:
                logger.warning("Could not extract results from eval_result, trying metric objects...")
                metric_names = ["answer_relevancy", "faithfulness", "contextual_precision"]
                for i, metric in enumerate(metrics):
                    metric_name = metric_names[i]
                    try:
                        score = float(metric.score) if metric.score is not None else 0.0
                        passed = metric.is_successful() if hasattr(metric, 'is_successful') else score >= DeepEvalConfig.METRIC_THRESHOLD
                        reason = getattr(metric, 'reason', None)

                        results[metric_name] = {
                            "score": score,
                            "passed": passed,
                            "reason": reason,
                            "description": metric_descriptions[metric_name]
                        }

                        status = "PASS" if passed else "FAIL"
                        logger.info(f"{metric_name}: {score:.4f} {status}")
                    except Exception as metric_error:
                        logger.error(f"{metric_name} processing failed: {str(metric_error)}")

        except Exception as e:
            logger.error(f"Parallel evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # Ensure we return error info if results are empty
            if not results:
                for metric_name in metric_descriptions:
                    results[metric_name] = {
                        "error": str(e),
                        "score": 0.0,
                        "passed": False,
                        "description": metric_descriptions[metric_name]
                    }

        logger.info("Evaluation complete!")

        return {
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "num_retrieved_chunks": len(retrieval_context),
            "metrics": results
        }


# At the bottom of deepeval.py

if __name__ == "__main__":
    from src.search import RAGSearch

    # Initialize RAG
    rag = RAGSearch(debug=False)

    # Initialize DeepEval (reference-based mode with ground truth)
    logger.info("Initializing DeepEval Service...")
    evaluator = DeepEvalService(rag)

    print("\n" + "=" * 80)
    print("INTERACTIVE RAG EVALUATION")
    print("=" * 80)
    print("\nEnter your queries to evaluate them against the RAG system.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            # Get user input
            query = input("\n📝 Enter your query: ").strip()

            if query.lower() in ['exit', 'quit', 'q']:
                print("\nExiting evaluation mode. Goodbye!")
                break

            if not query:
                print("Please enter a valid query.")
                continue

            print("\n" + "-" * 80)
            print(f"Evaluating: {query}\n")

            # Run evaluation
            result = evaluator.evaluate_query(query)

            # Display results
            print("\n" + "=" * 80)
            print("EVALUATION RESULTS")
            print("=" * 80)

            print(f"\n🔍 Query:\n{result['query']}")

            print(f"\n📄 RAG Answer:\n{result['answer']}")

            if result['expected_answer']:
                print(f"\n✅ Expected Answer (from ground truth):\n{result['expected_answer']}")
            else:
                print(f"\n⚠️  No matching ground truth found for this query")

            print(f"\n📊 Retrieved Chunks: {result['num_retrieved_chunks']}")

            print(f"\n📈 Metrics:")
            if 'error' in result.get('metrics', {}):
                print(f"   ❌ Error: {result['metrics']['error']}")
            else:
                for metric_name, metric_data in result.get('metrics', {}).items():
                    if isinstance(metric_data, dict):
                        if 'error' not in metric_data:
                            score = metric_data.get('score', 0)
                            passed = metric_data.get('passed', False)
                            status = "✓ PASS" if passed else "✗ FAIL"
                            reason = metric_data.get('reason')

                            print(f"\n   • {metric_name.replace('_', ' ').title()}")
                            print(f"     Score: {score:.4f} {status}")
                            if reason:
                                print(f"     Reason: {reason}")
                        else:
                            print(f"\n   • {metric_name.replace('_', ' ').title()}")
                            print(f"     Error: {metric_data.get('error')}")

            print("\n" + "=" * 80)

        except KeyboardInterrupt:
            print("\n\nEvaluation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nEvaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
