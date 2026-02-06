import os
import psycopg2
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import List, Dict, Tuple
from difflib import SequenceMatcher

# Handle imports for both module and direct execution
try:
    from src.vectorstore import PgVectorStore
    from src.config import DatabaseConfig, LLMConfig, SearchConfig, ConversationConfig
    from src.logger import get_search_logger
except ModuleNotFoundError:
    from vectorstore import PgVectorStore
    from config import DatabaseConfig, LLMConfig, SearchConfig, ConversationConfig
    from logger import get_search_logger

load_dotenv()

# Initialize logger
logger = get_search_logger()


# =====================================================================
# DATABASE HELPER (uses centralized config)
# =====================================================================

def get_db_connection():
    """Get a PostgreSQL connection using centralized config"""
    return psycopg2.connect(**DatabaseConfig.get_connection_params())


# =====================================================================
# COMPANY EXTRACTION UTILITIES
# =====================================================================

def lookup_employee_company(employee_name: str) -> Tuple[int, str, str]:
    """
    Look up an employee in the database to find their company.

    Returns:
        (company_id, company_name, employee_full_name) if found
        (None, None, None) if not found

    Checks both internal and external employees.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        name_lower = employee_name.lower()

        # Try external employees first
        cur.execute("""
            SELECT e.company_id, c.company_name, e.name
            FROM external_employees e
            JOIN companies c ON e.company_id = c.company_id
            WHERE LOWER(e.name) LIKE %s OR LOWER(e.name) = %s
            LIMIT 1
        """, (f"%{name_lower}%", name_lower))

        result = cur.fetchone()
        if result:
            return result[0], result[1], result[2]

        # Try internal employees
        cur.execute("""
            SELECT 1, 'Knoccs Internal', i.name
            FROM internal_employees i
            WHERE LOWER(i.name) LIKE %s OR LOWER(i.name) = %s
            LIMIT 1
        """, (f"%{name_lower}%", name_lower))

        result = cur.fetchone()
        if result:
            return result[0], result[1], result[2]

        return None, None, None

    except Exception as e:
        logger.error(f"Failed to lookup employee: {e}")
        return None, None, None
    finally:
        cur.close()
        conn.close()


def load_companies_map() -> Dict[str, int]:
    """
    Load all companies from database into a name -> id mapping.
    Handles variations: "Pulse Unity", "pulse unity", "PulseUnity", etc.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT company_id, company_name FROM companies ORDER BY company_id")
        rows = cur.fetchall()

        company_map = {}
        for company_id, company_name in rows:
            # Add multiple variations of the name
            company_map[company_name.lower()] = company_id  # "pulse unity"
            company_map[company_name.lower().replace(' ', '')] = company_id  # "pulseunity"
            company_map[company_name] = company_id  # Original case

        return company_map
    finally:
        cur.close()
        conn.close()


def extract_company_from_query(query: str, company_map: Dict[str, int]) -> Tuple[int, str]:
    """
    Extract company name from query using string matching.

    Returns:
        (company_id, company_name) if found
        (None, None) if not found

    Examples:
        "What is Pulse Unity's main business?" → (6, "Pulse Unity")
        "Tell me about vantage nexus" → (7, "Vantage Nexus")
    """
    query_lower = query.lower()
    query_no_space = query_lower.replace(' ', '')

    best_match = None
    best_match_id = None
    best_ratio = 0

    # Try to find exact or close matches
    for company_name, company_id in company_map.items():
        company_lower = company_name.lower()
        company_no_space = company_lower.replace(' ', '')

        # Exact match (case-insensitive)
        if company_lower in query_lower or company_no_space in query_no_space:
            return company_id, company_name

        # Fuzzy match - find the best partial match
        ratio = SequenceMatcher(None, query_lower, company_lower).ratio()
        if ratio > best_ratio and ratio > SearchConfig.COMPANY_MATCH_THRESHOLD:
            best_ratio = ratio
            best_match = company_name
            best_match_id = company_id

    if best_match_id is not None:
        return best_match_id, best_match

    return None, None


def extract_employee_name_from_query(query: str) -> str:
    """
    Extract potential employee name from query.
    Looks for patterns like "Which company does [NAME] work at?"

    Returns the potential name or None
    """
    import re

    # Pattern: "does [Name] work" or "works at" or similar
    patterns = [
        r'does\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+work',
        r'is\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:work|employ)',
        r'who\s+is\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'find\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1)

    return None


# =====================================================================
# RAG SEARCH CLASS
# =====================================================================

class RAGSearch:
    def __init__(self, embedding_model: str = None, llm_model: str = None,
                 use_query_expansion: bool = False,
                 debug: bool = False, vectorstore: PgVectorStore = None):

        # Use config defaults if not specified
        if embedding_model is None:
            from src.config import EmbeddingConfig
            embedding_model = EmbeddingConfig.MODEL

        if llm_model is None:
            llm_model = LLMConfig.MODEL

        if vectorstore:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = PgVectorStore(embedding_model=embedding_model, use_reranker=True)

        self.llm = ChatOpenAI(
            api_key=LLMConfig.OPENAI_API_KEY,
            model=llm_model,
            temperature=LLMConfig.TEMPERATURE
        )
        self.use_query_expansion = use_query_expansion
        self.debug = debug

        # Load companies map once on initialization
        self.company_map = load_companies_map()

        logger.info(f"LLM initialized: {llm_model}")
        logger.info(f"Loaded {len(self.company_map)} companies for extraction")

    def search_and_summarize(self, query: str, top_k: int = None, initial_k: int = None,
                             company_id: int = None, conversation_history: List[Dict] = None) -> Tuple[str, List[Dict]]:
        """
        Search and summarize with automatic company extraction and conversation history.

        Args:
            query: User's question
            top_k: Number of final results (default from config)
            initial_k: Number of initial results before reranking (default from config)
            company_id: Optional company_id. If not provided, extracted from query.
            conversation_history: List of previous messages

        Returns:
            (response_text, updated_conversation_history)
        """
        # Use config defaults
        if top_k is None:
            top_k = SearchConfig.TOP_K
        if initial_k is None:
            initial_k = SearchConfig.INITIAL_K

        if conversation_history is None:
            conversation_history = []

        query = query.strip()
        if not query:
            return "Please ask a question.", conversation_history

        # Step 0: Extract company from query if not provided
        if company_id is None:
            company_id, company_name = extract_company_from_query(query, self.company_map)

            # If no company found, try employee lookup
            if company_id is None:
                employee_name = extract_employee_name_from_query(query)
                if employee_name:
                    company_id, company_name, emp_name = lookup_employee_company(employee_name)
                    if company_id and self.debug:
                        logger.debug(f"Found employee {emp_name} from company: {company_name} (ID: {company_id})")

            # Check conversation history for context
            if company_id is None and conversation_history:
                if self.debug:
                    logger.debug(f"No company in query, checking history ({len(conversation_history)} messages)")

                for prev_msg in reversed(conversation_history):
                    prev_content = prev_msg.get("content", "")
                    found_company_id, found_company_name = extract_company_from_query(prev_content, self.company_map)
                    if found_company_id:
                        company_id = found_company_id
                        company_name = found_company_name
                        if self.debug:
                            logger.debug(f"Found company in history: {company_name} (ID: {company_id})")
                        break

            if company_id is None:
                return "I couldn't determine which company this question is about.", conversation_history

            if self.debug:
                logger.debug(f"Extracted company: {company_name} (ID: {company_id})")

        # Step 1: Manage conversation history
        conversation_context, summary_text = self._manage_conversation_history(conversation_history)

        # Step 2: Query expansion (optional)
        queries = [query]

        # Step 3: Retrieve and deduplicate
        all_results = []
        seen_chunk_ids = set()

        for q in queries:
            results = self.vectorstore.query(q, company_id=company_id, top_k=top_k, initial_k=initial_k, use_hybrid=True)
            if self.debug:
                logger.debug(f"Query returned {len(results)} results from company_id={company_id}")

            for r in results:
                chunk_id = r.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    all_results.append(r)
                    seen_chunk_ids.add(chunk_id)

        if self.debug:
            logger.debug(f"Total unique results after deduplication: {len(all_results)}")

        # Sort by score
        all_results.sort(
            key=lambda x: x.get("rerank_score", -x.get("distance", 999)),
            reverse=True
        )

        # Take top results
        all_results = all_results[:top_k]

        if self.debug and all_results:
            logger.debug(f"Top result score: {all_results[0].get('rerank_score', all_results[0].get('distance'))}")
            logger.debug(f"Top result preview: {all_results[0]['metadata'].get('text', '')[:200]}...")

        if not all_results:
            response = "I couldn't find any relevant documents for your query in this company's knowledge base."
        else:
            # Step 4: Build context with metadata
            context_parts = []
            for i, r in enumerate(all_results, 1):
                meta = r["metadata"]
                text = meta.get("text", "")
                source = meta.get("source", "unknown")
                doc_type = meta.get("document_type", "unknown")

                score_info = ""
                if self.debug:
                    score = r.get("rerank_score", r.get("distance", "N/A"))
                    score_info = f" [Score: {score:.3f}]"

                context_parts.append(
                    f"Source: {source} | Type: {doc_type}{score_info}\n{text}"
                )

            context = "\n\n".join(context_parts)

            # Step 5: Generate response
            prompt = f"""You are Knoccs AI Assistant, helping staff find information from client documents, agreements, and communications.

**Guidelines:**
- Answer based strictly on the provided documents for this client's knowledge base
- Do NOT use phrases like "According to Document 1" or "Document 2 states"
- If information isn't in the documents, say "I couldn't find this information in your available documents"
- For agreements, contracts, or policies, mention the specific document name at the end if relevant
- Write naturally as if you're an internal knowledge assistant familiar with this client's business
- Present information clearly and professionally for business use
- If multiple documents contain relevant information, synthesize them coherently into a single answer
- Be consistent with previous conversation context when relevant
- Maintain strict data isolation: only reference information from the current client's knowledge base
- Do not acknowledge or reference other clients' data or knowledge bases
- If a query seems to ask for cross-company information, clarify that you can only provide information about this client

**Conversation Context:**
{summary_text}

**Current Query:** {query}

**Retrieved Documents:**
{context}

**Response:**"""

            try:
                response_obj = self.llm.invoke(prompt)
                response = response_obj.content
            except Exception as e:
                logger.error(f"LLM generation failed: {str(e)}")
                response = "I encountered an error while generating a response. Please try again."

        # Step 6: Update conversation history
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": response})

        return response, conversation_history

    def _manage_conversation_history(self, conversation_history: List[Dict]) -> Tuple[List[Dict], str]:
        """
        Manage conversation history by keeping recent messages and summarizing older ones.

        Returns:
            (recent_messages, summary_text)
        """
        if not conversation_history:
            return [], ""

        max_recent = ConversationConfig.MAX_RECENT_MESSAGES

        if len(conversation_history) <= max_recent:
            return conversation_history, ""

        # Split into recent and old
        old_messages = conversation_history[:-max_recent]
        recent_messages = conversation_history[-max_recent:]

        # Summarize old messages
        summary = self._summarize_messages(old_messages)

        return recent_messages, summary

    def _summarize_messages(self, messages: List[Dict]) -> str:
        """Summarize old messages using LLM."""
        if not messages:
            return ""

        conversation_text = ""
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            conversation_text += f"{role}: {content}\n\n"

        prompt = f"""Summarize the following conversation in 2-3 sentences. Focus on:
- Key entities (company names, people, dates)
- Main topics discussed
- Important decisions or findings

Conversation:
{conversation_text}

Summary:"""

        try:
            response_obj = self.llm.invoke(prompt)
            return "Earlier in this conversation: " + response_obj.content
        except Exception as e:
            if self.debug:
                logger.debug(f"LLM summarization failed: {e}. Using fallback.")
            return ""

    def search_only(self, query: str, top_k: int = None, company_id: int = None) -> List[Dict]:
        """Return raw search results without LLM summarization."""
        if top_k is None:
            top_k = SearchConfig.TOP_K

        if company_id is None:
            company_id, _ = extract_company_from_query(query, self.company_map)
            if company_id is None:
                return []

        return self.vectorstore.query(query, company_id=company_id, top_k=top_k, initial_k=top_k * 4)

    def get_stats(self, company_id: int = None) -> Dict:
        """Get statistics about the knowledge base."""
        return self.vectorstore.get_stats(company_id=company_id)


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch(debug=True, use_query_expansion=False)

    print("Knowledge base stats:", rag_search.get_stats())

    query = "What is Pulse Unity's main business focus?"
    print(f"\n{'=' * 80}")
    print(f"QUERY: {query}")
    print('=' * 80)

    summary, _ = rag_search.search_and_summarize(query, top_k=15, initial_k=50)
    print("\nANSWER:")
    print(summary)
