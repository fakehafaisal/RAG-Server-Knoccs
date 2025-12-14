# """
# DeepEval wrapper service for optional DeepEval metrics.
# This file imports external `deepeval` package lazily and raises a clear error
# if it's not installed so the API endpoint can return a helpful message.
# """
# import os
# from dotenv import load_dotenv
# from src.search import RAGSearch
# from typing import Any, TYPE_CHECKING, Dict

# load_dotenv()

# if TYPE_CHECKING:
#     from deepeval.test_case import LLMTestCase  # type: ignore
#     from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric  # type: ignore


# class DeepEvalService:
#     def __init__(self, rag_search: RAGSearch):
#         self.rag_search = rag_search
        
#         # Check for OpenAI API key
#         self.openai_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_key:
#             raise ValueError(
#                 "OPENAI_API_KEY not found in environment variables. "
#                 "DeepEval requires OpenAI API to evaluate answers. "
#                 "Add OPENAI_API_KEY=sk-... to your .env file."
#             )
        
#         try:
#             # Lazy import external dependency and bind to instance
#             from deepeval.metrics import (
#                 AnswerRelevancyMetric, 
#                 FaithfulnessMetric, 
#                 ContextualRelevancyMetric
#             )
#             from deepeval.test_case import LLMTestCase
            
#             # Set API key for deepeval
#             os.environ["OPENAI_API_KEY"] = self.openai_key
            
#         except ImportError as e:
#             raise ImportError(
#                 "DeepEval package not installed. Install it with: pip install deepeval"
#             ) from e

#         self.AnswerRelevancyMetric = AnswerRelevancyMetric
#         self.FaithfulnessMetric = FaithfulnessMetric
#         self.ContextualRelevancyMetric = ContextualRelevancyMetric
#         self.LLMTestCase = LLMTestCase

#     def evaluate_query(self, query: str, expected_answer: str = None) -> Dict:
#         """Evaluate a query dynamically using the live RAG KB"""

#         # Step 1: Retrieve results and summary
#         raw_results = self.rag_search.search_only(query, top_k=15)
#         answer = self.rag_search.search_and_summarize(query, top_k=15)
#         retrieval_context = [r["metadata"]["text"] for r in raw_results]

#         # Validate we got results
#         if not retrieval_context:
#             return {
#                 "query": query,
#                 "answer": answer,
#                 "num_retrieved_chunks": 0,
#                 "metrics": {
#                     "error": "No chunks retrieved - cannot evaluate"
#                 }
#             }

#         # Step 2: Create test case
#         test_case = self.LLMTestCase(
#             input=query,
#             actual_output=answer,
#             retrieval_context=retrieval_context,
#             expected_output=expected_answer
#         )

#         results = {}

#         # Step 3: Run metrics with better error handling
#         print(f"[INFO] Running DeepEval metrics for query: {query[:50]}...")
        
#         # Answer Relevancy
#         try:
#             print("[INFO] Evaluating Answer Relevancy...")
#             metric = self.AnswerRelevancyMetric(
#                 threshold=0.7,
#                 model="gpt-4o-mini",  # Use cheaper model for evaluation
#                 include_reason=True
#             )
#             metric.measure(test_case)
#             results["answer_relevancy"] = {
#                 "score": float(metric.score) if metric.score is not None else None,
#                 "passed": metric.is_successful(),
#                 "reason": metric.reason if hasattr(metric, 'reason') else None
#             }
#             print(f"[SUCCESS] Answer Relevancy: {metric.score:.2f}")
#         except Exception as e:
#             print(f"[ERROR] Answer Relevancy failed: {str(e)}")
#             results["answer_relevancy"] = {"error": str(e), "score": None, "passed": False}

#         # Faithfulness
#         try:
#             print("[INFO] Evaluating Faithfulness...")
#             metric = self.FaithfulnessMetric(
#                 threshold=0.7,
#                 model="gpt-4o-mini",
#                 include_reason=True
#             )
#             metric.measure(test_case)
#             results["faithfulness"] = {
#                 "score": float(metric.score) if metric.score is not None else None,
#                 "passed": metric.is_successful(),
#                 "reason": metric.reason if hasattr(metric, 'reason') else None
#             }
#             print(f"[SUCCESS] Faithfulness: {metric.score:.2f}")
#         except Exception as e:
#             print(f"[ERROR] Faithfulness failed: {str(e)}")
#             results["faithfulness"] = {"error": str(e), "score": None, "passed": False}

#         # Contextual Relevancy
#         try:
#             print("[INFO] Evaluating Contextual Relevancy...")
#             metric = self.ContextualRelevancyMetric(
#                 threshold=0.7,
#                 model="gpt-4o-mini",
#                 include_reason=True
#             )
#             metric.measure(test_case)
#             results["contextual_relevancy"] = {
#                 "score": float(metric.score) if metric.score is not None else None,
#                 "passed": metric.is_successful(),
#                 "reason": metric.reason if hasattr(metric, 'reason') else None
#             }
#             print(f"[SUCCESS] Contextual Relevancy: {metric.score:.2f}")
#         except Exception as e:
#             print(f"[ERROR] Contextual Relevancy failed: {str(e)}")
#             results["contextual_relevancy"] = {"error": str(e), "score": None, "passed": False}

#         return {
#             "query": query,
#             "answer": answer,
#             "num_retrieved_chunks": len(retrieval_context),
#             "metrics": results
#         }


"""
DeepEval wrapper service compatible with DeepEval 3.7.5
Uses direct Groq integration without LiteLLM dependency issues
"""
import os
from dotenv import load_dotenv
from src.search import RAGSearch
from typing import Dict
from langchain_groq import ChatGroq

load_dotenv()


class DeepEvalService:
    def __init__(self, rag_search: RAGSearch, use_groq: bool = True):
        """
        Initialize DeepEval with Groq or OpenAI support
        
        Args:
            rag_search: Your RAG search instance
            use_groq: If True, uses Groq. If False, uses OpenAI
        """
        self.rag_search = rag_search
        self.use_groq = use_groq
        
        try:
            # Import DeepEval components
            from deepeval.metrics import (
                AnswerRelevancyMetric, 
                FaithfulnessMetric, 
                ContextualRelevancyMetric
            )
            from deepeval.test_case import LLMTestCase
            
            self.AnswerRelevancyMetric = AnswerRelevancyMetric
            self.FaithfulnessMetric = FaithfulnessMetric
            self.ContextualRelevancyMetric = ContextualRelevancyMetric
            self.LLMTestCase = LLMTestCase
            
        except ImportError as e:
            raise ImportError(
                "DeepEval package not installed. Install it with: pip install deepeval"
            ) from e
        
        # Setup evaluation model
        if use_groq:
            self._setup_groq()
        else:
            self._setup_openai()
    
    def _setup_groq(self):
        """
        Setup Groq for evaluation using custom model wrapper
        Compatible with DeepEval 3.7.5
        """
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables. "
                "Add GROQ_API_KEY=... to your .env file."
            )
        
        try:
            from deepeval.models.base_model import DeepEvalBaseLLM
            
            class GroqModelWrapper(DeepEvalBaseLLM):
                """Custom Groq wrapper for DeepEval 3.x"""
                
                def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
                    self.model_name = model_name
                    self.groq_model = ChatGroq(
                        groq_api_key=groq_key,
                        model_name=model_name,
                        temperature=0.0,
                        max_tokens=2000
                    )
                
                def load_model(self):
                    """Load the Groq model"""
                    return self.groq_model
                
                def generate(self, prompt: str) -> str:
                    """Generate response from Groq"""
                    try:
                        response = self.groq_model.invoke(prompt)
                        return response.content
                    except Exception as e:
                        print(f"[ERROR] Groq generation failed: {e}")
                        raise
                
                async def a_generate(self, prompt: str) -> str:
                    """Async generate (uses sync for now)"""
                    return self.generate(prompt)
                
                def get_model_name(self) -> str:
                    """Return model name"""
                    return self.model_name
            
            # Create the wrapper
            self.evaluation_model = GroqModelWrapper()
            print(f"[INFO] Using Groq (llama-3.3-70b-versatile) for DeepEval metrics")
            
        except Exception as e:
            print(f"[ERROR] Failed to setup Groq: {e}")
            raise
    
    def _setup_openai(self):
        """Setup OpenAI as the evaluation LLM (fallback)"""
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Add OPENAI_API_KEY=sk-... to your .env file."
            )
        
        os.environ["OPENAI_API_KEY"] = openai_key
        # For OpenAI, DeepEval can use default without custom wrapper
        self.evaluation_model = "gpt-4o-mini"
        print(f"[INFO] Using OpenAI (gpt-4o-mini) for DeepEval metrics")

    def evaluate_query(self, query: str, expected_answer: str = None) -> Dict:
        """Evaluate a query dynamically using the live RAG KB"""

        print(f"[INFO] Starting evaluation for query: {query[:50]}...")
        
        # Step 1: Retrieve results and summary
        try:
            raw_results = self.rag_search.search_only(query, top_k=15)
            answer = self.rag_search.search_and_summarize(query, top_k=15)
            retrieval_context = [r["metadata"]["text"] for r in raw_results]
        except Exception as e:
            print(f"[ERROR] Failed to retrieve results: {e}")
            return {
                "query": query,
                "answer": f"Error: {str(e)}",
                "num_retrieved_chunks": 0,
                "metrics": {"error": f"Retrieval failed: {str(e)}"}
            }

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

        print(f"[INFO] Retrieved {len(retrieval_context)} chunks")
        print(f"[INFO] Answer length: {len(answer)} characters")

        # Step 2: Create test case
        test_case = self.LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=retrieval_context,
            expected_output=expected_answer
        )

        results = {}

        # Step 3: Run metrics with detailed error handling
        print(f"[INFO] Running DeepEval metrics...")
        
        # Answer Relevancy
        try:
            print("[INFO] → Evaluating Answer Relevancy...")
            metric = self.AnswerRelevancyMetric(
                threshold=0.7,
                model=self.evaluation_model,
                include_reason=True
            )
            metric.measure(test_case)
            
            score = float(metric.score) if metric.score is not None else 0.0
            results["answer_relevancy"] = {
                "score": score,
                "passed": metric.is_successful(),
                "reason": getattr(metric, 'reason', None)
            }
            print(f"[SUCCESS] Answer Relevancy: {score:.4f} {'✓' if metric.is_successful() else '✗'}")
            
        except Exception as e:
            print(f"[ERROR] Answer Relevancy failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results["answer_relevancy"] = {
                "error": str(e), 
                "score": 0.0, 
                "passed": False
            }

        # Faithfulness
        try:
            print("[INFO] → Evaluating Faithfulness...")
            metric = self.FaithfulnessMetric(
                threshold=0.7,
                model=self.evaluation_model,
                include_reason=True
            )
            metric.measure(test_case)
            
            score = float(metric.score) if metric.score is not None else 0.0
            results["faithfulness"] = {
                "score": score,
                "passed": metric.is_successful(),
                "reason": getattr(metric, 'reason', None)
            }
            print(f"[SUCCESS] Faithfulness: {score:.4f} {'✓' if metric.is_successful() else '✗'}")
            
        except Exception as e:
            print(f"[ERROR] Faithfulness failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results["faithfulness"] = {
                "error": str(e), 
                "score": 0.0, 
                "passed": False
            }

        # Contextual Relevancy
        try:
            print("[INFO] → Evaluating Contextual Relevancy...")
            metric = self.ContextualRelevancyMetric(
                threshold=0.7,
                model=self.evaluation_model,
                include_reason=True
            )
            metric.measure(test_case)
            
            score = float(metric.score) if metric.score is not None else 0.0
            results["contextual_relevancy"] = {
                "score": score,
                "passed": metric.is_successful(),
                "reason": getattr(metric, 'reason', None)
            }
            print(f"[SUCCESS] Contextual Relevancy: {score:.4f} {'✓' if metric.is_successful() else '✗'}")
            
        except Exception as e:
            print(f"[ERROR] Contextual Relevancy failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results["contextual_relevancy"] = {
                "error": str(e), 
                "score": 0.0, 
                "passed": False
            }

        print(f"[INFO] Evaluation complete!")
        
        return {
            "query": query,
            "answer": answer,
            "num_retrieved_chunks": len(retrieval_context),
            "metrics": results
        }


# Example usage
if __name__ == "__main__":
    from src.search import RAGSearch
    
    print("="*80)
    print("DEEPEVAL WITH GROQ TEST (DeepEval 3.7.5)")
    print("="*80)
    
    # Initialize RAG
    rag = RAGSearch(debug=False)
    
    # Initialize DeepEval with Groq
    print("\n[INFO] Initializing DeepEval with Groq...")
    evaluator = DeepEvalService(rag, use_groq=True)
    
    # Test query
    query = "What is the main topic of the documents?"
    print(f"\n[INFO] Testing query: {query}")
    
    result = evaluator.evaluate_query(query)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nQuery: {result['query']}")
    print(f"Answer: {result['answer'][:300]}...")
    print(f"Chunks Retrieved: {result['num_retrieved_chunks']}")
    
    print(f"\n{'Metric':<25} {'Score':<10} {'Status':<10}")
    print("-"*50)
    
    for metric_name, metric_data in result['metrics'].items():
        if 'error' in metric_data:
            print(f"{metric_name:<25} {'ERROR':<10} {'✗':<10}")
            print(f"  Error: {metric_data['error']}")
        else:
            score = metric_data.get('score', 0)
            status = "✓ PASS" if metric_data.get('passed') else "✗ FAIL"
            print(f"{metric_name:<25} {score:<10.4f} {status:<10}")
    
    print("\n" + "="*80)