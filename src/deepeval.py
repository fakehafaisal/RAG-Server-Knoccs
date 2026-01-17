import os
import json
from dotenv import load_dotenv
from typing import Dict, Optional
from difflib import SequenceMatcher

# Handle imports properly
try:
    from src.search import RAGSearch
except ModuleNotFoundError:
    from search import RAGSearch

load_dotenv()


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
            
            print("DeepEval initialized with OpenAI GPT-5")
            print(f"Running in reference-based mode with {len(self.ground_truth_data)} ground truth entries")
            
        except ImportError as e:
            raise ImportError(
                "DeepEval package not installed. Install it with: pip install deepeval"
            ) from e
    
    def _load_ground_truth(self, ground_truth_file: str):
        """Load ground truth data from JSON file"""
        try:
            if os.path.exists(ground_truth_file):
                with open(ground_truth_file, 'r') as f:
                    self.ground_truth_data = json.load(f)
                print(f"[INFO] Loaded {len(self.ground_truth_data)} ground truth entries from {ground_truth_file}")
            else:
                print(f"[WARNING] Ground truth file not found: {ground_truth_file}")
                self.ground_truth_data = []
        except Exception as e:
            print(f"[ERROR] Failed to load ground truth: {e}")
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
        threshold = 0.6  # Require 60% similarity
        
        for entry in self.ground_truth_data:
            gt_query = entry.get("query", "").lower()
            query_lower = query.lower()
            
            # Calculate similarity ratio
            ratio = SequenceMatcher(None, query_lower, gt_query).ratio()
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = entry
        
        if best_ratio >= threshold:
            print(f"[INFO] Found matching ground truth (similarity: {best_ratio:.2%})")
            return best_match
        
        print(f"[INFO] No matching ground truth found (best match: {best_ratio:.2%})")
        return None



    def evaluate_query(self, query: str) -> Dict:
        # Step 1: Retrieve results and generate answer
        try:
            raw_results = self.rag_search.search_only(query, top_k=15)
            # search_and_summarize now returns (answer, conversation_history) tuple
            answer, _ = self.rag_search.search_and_summarize(query, top_k=15, conversation_history=[])
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

        # Step 2: Find matching ground truth
        ground_truth_match = self._find_matching_ground_truth(query)
        expected_answer = None
        if ground_truth_match:
            expected_answer = ground_truth_match.get("expected_answer")
            print(f"[INFO] Using expected answer from ground truth")

        # Step 3: Create test case with ground truth (if available)
        test_case = self.LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=retrieval_context,
            expected_output=expected_answer  # Now includes expected output for reference-based metrics
        )

        results = {}

        # ============ METRIC 1: Answer Relevancy ============
        try:
            print("[INFO] → Evaluating Answer Relevancy...")
            metric = self.AnswerRelevancyMetric(
                model=self.openai_model, 
                threshold=0.7
            )
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
            print("[INFO] (This metric requires API calls - may take 20-30 seconds...)")
            
            metric = self.FaithfulnessMetric(
                model=self.openai_model, 
                threshold=0.7,
                async_mode=False
            )
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
            print("[INFO] (This metric requires API calls - may take 20-30 seconds...)")
            
            metric = self.ContextualPrecisionMetric(
                model=self.openai_model, 
                threshold=0.7
            )
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
            "expected_answer": expected_answer,  # Now includes ground truth expected answer
            "num_retrieved_chunks": len(retrieval_context),
            "metrics": results
        }


# At the bottom of deepeval.py

if __name__ == "__main__":
    from src.search import RAGSearch
    
    # Initialize RAG
    rag = RAGSearch(debug=False)
    
    # Initialize DeepEval (reference-based mode with ground truth)
    print("\n[INFO] Initializing DeepEval Service...")
    evaluator = DeepEvalService(rag)
    
    print("\n" + "="*80)
    print("INTERACTIVE RAG EVALUATION")
    print("="*80)
    print("\nEnter your queries to evaluate them against the RAG system.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            # Get user input
            query = input("\n📝 Enter your query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n[INFO] Exiting evaluation mode. Goodbye!")
                break
            
            if not query:
                print("[WARNING] Please enter a valid query.")
                continue
            
            print("\n" + "-"*80)
            print(f"Evaluating: {query}\n")
            
            # Run evaluation
            result = evaluator.evaluate_query(query)
            
            # Display results
            print("\n" + "="*80)
            print("EVALUATION RESULTS")
            print("="*80)
            
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
            
            print("\n" + "="*80)
            
        except KeyboardInterrupt:
            print("\n\n[INFO] Evaluation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] Evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()