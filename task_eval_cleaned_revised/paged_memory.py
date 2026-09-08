"""MemGPT-style memory system using Letta for LoCoMo evaluation."""

import os
import json
import logging
from typing import Dict, List, Any, Optional

from letta import create_client, LocalClient
from letta.schemas.memory import ChatMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NVIDIA API Configuration
# ---------------------------------------------------------------------------
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "http://localhost:8000/v1")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
LLM_MODEL = os.environ.get("NVIDIA_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


def create_memgpt_agent(
    agent_name: str,
    persona: str,
    human: str,
    model: str = LLM_MODEL,
    base_url: str = NVIDIA_BASE_URL,
    api_key: str = NVIDIA_API_KEY,
    embedding_model: str = "nvidia/nemotron-3-embed-1b",
    use_rag: bool = False,
) -> LocalClient:
    """
    Create a Letta (MemGPT-style) agent with proper memory configuration.

    Args:
        agent_name: Name for the agent
        persona: Initial persona description
        human: Initial human description
        model: LLM model identifier (from your NIM deployment)
        base_url: NVIDIA NIM base URL
        api_key: NVIDIA API key
        embedding_model: Embedding model for archival memory
        use_rag: Whether to use RAG-based retrieval (True) or MemGPT-style
                core+archival memory (False)

    Returns:
        Configured Letta LocalClient instance
    """
    # Initialize Letta client
    client = create_client()

    # Configure the agent's core memory (in-context)
    memory = ChatMemory(
        persona=persona,
        human=human,
        limit=2000,  # default character limit per memory section
    )

    # Create agent with memory configuration
    agent_state = client.create_agent(
        name=agent_name,
        memory=memory,
        model=model,  # LLM model from NIM
        embedding_model=embedding_model,  # for archival memory
        base_url=base_url,
        api_key=api_key,
    )

    return client, agent_state


def initialize_memory_from_conversation(
    client: LocalClient,
    agent_state: Any,
    conversation: Dict[str, Any],
) -> None:
    """
    Populate the agent's memory from a full LoCoMo conversation.

    This involves:
    1. Writing session summaries to core memory (if they exist)
    2. Writing conversation turns to recall memory
    3. Writing long-term facts to archival memory

    Args:
        client: Letta client
        agent_state: Agent state returned by create_agent
        conversation: LoCoMo conversation dictionary
    """
    session_nums = [
        int(k.split("_")[-1])
        for k in conversation.keys()
        if k.startswith("session_") and not k.endswith("_date_time")
    ]

    # 1. Populate recall memory with conversation history
    # (Letta automatically stores tool calls and messages in recall memory)
    for i in session_nums:
        key = f"session_{i}"
        if key not in conversation:
            continue

        for turn in conversation[key]:
            message = f'{turn["speaker"]} said: "{turn["text"]}"'
            if "blip_caption" in turn:
                message += f' [shares {turn["blip_caption"]}]'

            # Send as a user message (which gets stored in recall memory)
            client.send_message(
                agent_id=agent_state.id,
                message=message,
                role="user",
            )

    # 2. Populate core memory with critical facts (if summaries exist)
    # You can manually update core memory with important information
    for i in session_nums:
        summary_key = f"session_{i}_summary"
        if summary_key in conversation:
            summary_text = conversation[summary_key]
            # Append to core memory (persona or human section)
            # Modify this based on your needs
            client.core_memory_append(
                agent_id=agent_state.id,
                name="human",  # or "persona" depending on what you want to store
                content=summary_text,
            )

    # 3. Populate archival memory with long-term facts
    # (This is where you'd store facts that don't fit in core memory)
    # The agent can search this via archival_memory_search tool
    for i in session_nums:
        summary_key = f"session_{i}_summary"
        if summary_key in conversation:
            client.archival_memory_insert(
                agent_id=agent_state.id,
                text=conversation[summary_key],
            )


def retrieve_context_memgpt(
    client: LocalClient,
    agent_state: Any,
    question: str,
    top_k: int = 5,
) -> str:
    """
    Use MemGPT-style memory retrieval to find relevant context.

    The agent uses its tools (archival_memory_search, recall_memory_search)
    to retrieve information relevant to the question.

    Args:
        client: Letta client
        agent_state: Agent state
        question: The question to answer
        top_k: Maximum number of context chunks to retrieve

    Returns:
        Retrieved context string
    """
    # Use the agent's memory search to retrieve relevant context
    # For RAG-style retrieval, you can use the archival_memory_search tool
    # directly instead of letting the agent do it autonomously

    # Option 1: Let the agent decide (agentic retrieval)
    # (This is the "proper" MemGPT way - the agent decides what to retrieve)
    response = client.send_message(
        agent_id=agent_state.id,
        message=f"Question: {question}\nPlease search your memory and answer.",
        role="user",
    )
    return response.messages[-1].content

    # Option 2: Direct archival memory search (RAG-style)
    # results = client.archival_memory_search(
    #     agent_id=agent_state.id,
    #     query=question,
    #     top_k=top_k,
    # )
    # return "\n".join([r.text for r in results])


def build_memgpt_context(
    client: LocalClient,
    agent_state: Any,
    question: str,
    conversation: Dict[str, Any],
    top_k: int = 5,
    use_archival: bool = True,
) -> str:
    """
    Build a context string using MemGPT-style memory.

    This combines:
    1. Agent's core memory (always in context)
    2. Retrieved archival memory (via vector search)
    3. Retrieved recall memory (conversation history)

    Args:
        client: Letta client
        agent_state: Agent state
        question: Question to answer
        conversation: LoCoMo conversation (for direct recall if needed)
        top_k: Number of archival chunks to retrieve
        use_archival: Whether to use archival memory search

    Returns:
        Combined context string
    """
    context_parts = []

    # 1. Core memory (always present)
    core_memory = client.get_core_memory(agent_id=agent_state.id)
    if core_memory:
        context_parts.append(f"CORE MEMORY:\n{core_memory}")

    # 2. Archival memory search (vector retrieval)
    if use_archival:
        try:
            results = client.archival_memory_search(
                agent_id=agent_state.id,
                query=question,
                top_k=top_k,
            )
            if results:
                archival_context = "\n".join([r.text for r in results])
                context_parts.append(
                    f"ARCHIVAL MEMORY (retrieved):\n{archival_context}"
                )
        except Exception as e:
            logger.warning(f"Archival memory search failed: {e}")

    # 3. Recall memory search (conversation history)
    try:
        recall_results = client.recall_memory_search(
            agent_id=agent_state.id,
            query=question,
            top_k=top_k,
        )
        if recall_results:
            recall_context = "\n".join([r.text for r in recall_results])
            context_parts.append(f"RECALL MEMORY (retrieved):\n{recall_context}")
    except Exception as e:
        logger.warning(f"Recall memory search failed: {e}")

    return "\n\n".join(context_parts)
