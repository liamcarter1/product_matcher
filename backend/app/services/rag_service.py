"""
RAG Service with LangGraph for Danfoss product queries.
Implements a multi-step retrieval and response generation pipeline.
"""

import re
import uuid
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from ..config import get_settings
from .pinecone_service import PineconeService
from .confidence import ConfidenceScorer, ConfidenceResult
from .skill_loader import get_part_lookup_context, get_spec_query_context


class QueryType:
    """Query type classifications."""
    PART_LOOKUP = "part_lookup"
    SPECIFICATION_QUERY = "specification_query"
    GENERAL_QUESTION = "general_question"


class RAGState(TypedDict):
    """State for the RAG graph."""
    messages: List[Any]
    query: str
    query_type: str
    context: List[Dict[str, Any]]
    confidence: Optional[ConfidenceResult]
    response: str
    session_id: str
    sources: List[Dict[str, Any]]


class RAGService:
    """
    RAG service using LangGraph for structured query processing.

    Flow:
    1. Query Classification - Detect intent (part lookup, spec query, general)
    2. Retrieval - Fetch relevant documents from Pinecone
    3. Confidence Scoring - Calculate response confidence
    4. Response Generation - Generate professional response with LLM
    """

    # Session storage (in production, use Redis or similar)
    _sessions: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(self):
        """Initialize the RAG service."""
        self.settings = get_settings()

        # Initialize components
        self.llm = ChatOpenAI(
            model=self.settings.openai_model,
            api_key=self.settings.openai_api_key,
            temperature=0.3  # Lower temperature for more consistent responses
        )

        self.pinecone = PineconeService()
        self.confidence_scorer = ConfidenceScorer()

        # Build the LangGraph workflow
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("score_confidence", self._score_confidence)
        workflow.add_node("generate_response", self._generate_response)

        # Set entry point
        workflow.set_entry_point("classify_query")

        # Add edges
        workflow.add_edge("classify_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "score_confidence")
        workflow.add_edge("score_confidence", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def _classify_query(self, state: RAGState) -> RAGState:
        """
        Classify the query type to determine retrieval strategy.

        Types:
        - part_lookup: Looking up Danfoss equivalent for competitor part
        - specification_query: Asking about product specifications
        - general_question: General product/installation questions
        """
        query = state["query"].lower()

        # Part lookup patterns
        part_lookup_patterns = [
            r'replace\w*\s+\w+\s+part',
            r'equivalent\s+to',
            r'cross[\s-]?ref',
            r'what\s+danfoss\s+part',
            r'danfoss\s+equivalent',
            r'substitute\s+for',
            r'compatible\s+with',
            r'replaces?\s+\w+[\s-]?\d+',
            r'convert\s+from',
        ]

        # Specification patterns
        spec_patterns = [
            r'voltage\s+rating',
            r'current\s+rating',
            r'power\s+consumption',
            r'dimension',
            r'specification',
            r'spec\s+sheet',
            r'technical\s+data',
            r'what\s+is\s+the\s+\w+\s+of',
            r'operating\s+temp',
        ]

        # Check patterns
        for pattern in part_lookup_patterns:
            if re.search(pattern, query):
                state["query_type"] = QueryType.PART_LOOKUP
                return state

        for pattern in spec_patterns:
            if re.search(pattern, query):
                state["query_type"] = QueryType.SPECIFICATION_QUERY
                return state

        # Check if query contains a part number pattern (alphanumeric with dashes)
        part_number_pattern = r'\b[A-Za-z]{1,4}[\s-]?\d{2,}[\s-]?\w*\b'
        if re.search(part_number_pattern, query):
            # If it mentions "Danfoss" it's likely a spec query
            if "danfoss" in query:
                state["query_type"] = QueryType.SPECIFICATION_QUERY
            else:
                state["query_type"] = QueryType.PART_LOOKUP
            return state

        state["query_type"] = QueryType.GENERAL_QUESTION
        return state

    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents based on query type."""
        query = state["query"]
        query_type = state["query_type"]

        # Extract potential part numbers from query
        part_numbers = self._extract_part_numbers(query)

        if query_type == QueryType.PART_LOOKUP and part_numbers:
            # Use part number specific search
            results = []
            for part_num in part_numbers[:2]:  # Limit to first 2 part numbers
                part_results = self.pinecone.query_by_part_number(
                    part_num,
                    is_competitor_part=True,
                    top_k=3
                )
                results.extend(part_results)

            # Also do semantic search
            semantic_results = self.pinecone.query(
                query,
                namespace=PineconeService.NAMESPACE_PRODUCTS,
                top_k=3
            )
            results.extend(semantic_results)

            # Deduplicate by ID
            seen_ids = set()
            unique_results = []
            for r in results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    unique_results.append(r)

            # Sort by score
            unique_results.sort(key=lambda x: x["score"], reverse=True)
            state["context"] = unique_results[:5]

        elif query_type == QueryType.SPECIFICATION_QUERY:
            # Search primarily in products namespace
            results = self.pinecone.query(
                query,
                top_k=5
            )
            state["context"] = results

        else:
            # General question - search all namespaces
            results = self.pinecone.query(
                query,
                top_k=5
            )
            state["context"] = results

        # Extract sources
        state["sources"] = [
            {
                "file": doc["metadata"].get("source_file", "Unknown"),
                "type": doc["metadata"].get("file_type", "Unknown"),
                "chunk_id": doc["id"]
            }
            for doc in state["context"]
        ]

        return state

    def _extract_part_numbers(self, query: str) -> List[str]:
        """Extract potential part numbers from a query."""
        # Pattern for typical part numbers: ABC-123, ABC123, 123-ABC, etc.
        patterns = [
            r'\b[A-Za-z]{1,5}[\s-]?\d{2,}[\s-]?\w*\b',  # ABC-123, ABC123
            r'\b\d{2,}[\s-]?[A-Za-z]{1,5}[\s-]?\w*\b',  # 123-ABC
            r'\b\d{4,}\b',  # Pure numeric codes
        ]

        part_numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            part_numbers.extend(matches)

        # Clean up and deduplicate
        cleaned = []
        for pn in part_numbers:
            pn_clean = pn.strip().upper()
            if pn_clean and pn_clean not in cleaned:
                cleaned.append(pn_clean)

        return cleaned

    def _score_confidence(self, state: RAGState) -> RAGState:
        """Calculate confidence score for the retrieved context."""
        confidence = self.confidence_scorer.calculate_confidence(
            query=state["query"],
            retrieved_docs=state["context"]
        )
        state["confidence"] = confidence
        return state

    def _generate_response(self, state: RAGState) -> RAGState:
        """Generate the final response using the LLM."""
        query_type = state["query_type"]
        context = state["context"]
        confidence = state["confidence"]

        # Build context string
        context_str = self._format_context(context)

        # Get conversation history
        messages = state.get("messages", [])

        # Build system prompt based on query type
        system_prompt = self._get_system_prompt(query_type, confidence)

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])

        # Format messages for history
        history = []
        for msg in messages[-6:]:  # Last 6 messages for context
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))

        # Generate response
        chain = prompt | self.llm

        result = chain.invoke({
            "context": context_str,
            "query": state["query"],
            "history": history,
            "confidence_level": confidence.level.value if confidence else "unknown"
        })

        response = result.content

        # Add disclaimer if confidence is low
        if confidence and confidence.disclaimer:
            response += f"\n\n{confidence.disclaimer}"

        state["response"] = response
        return state

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format retrieved context for the prompt."""
        if not context:
            return "No relevant information found in the database."

        formatted = []
        for i, doc in enumerate(context, 1):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source_file", "Unknown")
            formatted.append(f"[Source {i}: {source}]\n{content}")

        return "\n\n".join(formatted)

    def _get_system_prompt(self, query_type: str, confidence: ConfidenceResult) -> str:
        """Get the appropriate system prompt based on query type."""
        base_prompt = """You are a professional technical assistant for Danfoss distributors.
Your role is to help distributors find Danfoss parts that replace competitor products and answer technical questions.

Guidelines:
- Be accurate and professional in your responses
- Only provide information that is supported by the context provided
- If you're not certain about something, say so
- When providing part number equivalents, be explicit about the Danfoss part number
- Include relevant specifications when available
- Do not make up information not present in the context

Context from product database:
{context}

Current confidence level: {confidence_level}
"""

        if query_type == QueryType.PART_LOOKUP:
            domain_ctx = get_part_lookup_context()
            domain_block = f"\n\nHydraulic valve domain knowledge:\n{domain_ctx}\n" if domain_ctx else ""
            return base_prompt + domain_block + """
For part lookup queries:
- Clearly state the Danfoss equivalent part number
- Mention the competitor brand and part being replaced
- When matching spool types across manufacturers, use the hydraulic function (flow path description) not the code name alone
- Include key specifications if available
- If no exact match is found, say so clearly
"""

        elif query_type == QueryType.SPECIFICATION_QUERY:
            domain_ctx = get_spec_query_context()
            domain_block = f"\n\nHydraulic valve domain knowledge:\n{domain_ctx}\n" if domain_ctx else ""
            return base_prompt + domain_block + """
For specification queries:
- Provide specific technical details requested
- Use proper units (voltage, current, dimensions, etc.)
- Use the unit normalisation rules above when comparing values across manufacturers
- Reference the source document when possible
"""

        else:
            return base_prompt + """

For general questions:
- Provide helpful and accurate information
- If the question is outside your knowledge, direct them to Danfoss technical support
- Be concise but thorough
"""

    async def query(
        self,
        message: str,
        session_id: Optional[str] = None,
        distributor_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.

        Args:
            message: User's query
            session_id: Session identifier for conversation continuity
            distributor_id: Optional distributor identifier

        Returns:
            Response dict with answer, confidence, and metadata
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Get or create session history
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        session_messages = self._sessions[session_id]

        # Add user message to history
        session_messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Initialize state
        initial_state: RAGState = {
            "messages": session_messages,
            "query": message,
            "query_type": "",
            "context": [],
            "confidence": None,
            "response": "",
            "session_id": session_id,
            "sources": []
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Add assistant response to history
        session_messages.append({
            "role": "assistant",
            "content": final_state["response"],
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": final_state["confidence"].score if final_state["confidence"] else None
        })

        # Keep only last 20 messages
        if len(session_messages) > 20:
            self._sessions[session_id] = session_messages[-20:]

        # Build response
        confidence = final_state["confidence"]

        return {
            "response": final_state["response"],
            "confidence": confidence.score if confidence else 0,
            "confidence_level": confidence.level.value if confidence else "low",
            "session_id": session_id,
            "sources": final_state["sources"],
            "query_type": final_state["query_type"],
            "disclaimer": confidence.disclaimer if confidence else None
        }

    def clear_session(self, session_id: str):
        """Clear a session's conversation history."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        return self._sessions.get(session_id, [])
