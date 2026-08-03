"""
agent.py
--------
A very simple LangGraph agent (built on top of LangChain) that writes a blog
post in three sequential steps:

    outline  ->  draft  ->  polish

Each step is a plain node function that calls a Groq-hosted LLM through
`langchain_groq.ChatGroq`. The graph's shared state is a small TypedDict
that gets passed from node to node and filled in along the way.

This file has no Streamlit code in it on purpose -- it's just the
"brain" of the app so it can be reused, tested, or imported elsewhere.
"""

from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END


# ---------------------------------------------------------------------------
# 1. Shared state that flows through the graph
# ---------------------------------------------------------------------------
class BlogState(TypedDict):
    topic: str          # what the user wants a blog about
    tone: str            # e.g. "friendly", "formal", "witty"
    length: str          # e.g. "short", "medium", "long"
    outline: str          # produced by the "outline" node
    draft: str            # produced by the "draft" node
    final_post: str        # produced by the "polish" node
    model: str            # which Groq model to use
    api_key: str           # the user's Groq API key


# ---------------------------------------------------------------------------
# 2. Small helper to build a ChatGroq client for a given node call
# ---------------------------------------------------------------------------
def _get_llm(state: BlogState, temperature: float = 0.7) -> ChatGroq:
    return ChatGroq(
        groq_api_key=state["api_key"],
        model=state["model"],
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# 3. Node functions -- each takes the state in, returns a partial state update
# ---------------------------------------------------------------------------
def create_outline(state: BlogState) -> dict:
    llm = _get_llm(state, temperature=0.6)
    system = SystemMessage(content=(
        "You are an expert content strategist. Produce a clear, "
        "well-structured blog outline with a working title, an intro hook, "
        "4-6 section headings, and a conclusion beat. Keep it concise."
    ))
    human = HumanMessage(content=(
        f"Topic: {state['topic']}\n"
        f"Tone: {state['tone']}\n"
        f"Target length: {state['length']}\n\n"
        "Write only the outline."
    ))
    response = llm.invoke([system, human])
    return {"outline": response.content}


def write_draft(state: BlogState) -> dict:
    llm = _get_llm(state, temperature=0.8)
    system = SystemMessage(content=(
        "You are a skilled blog writer. Turn the given outline into a full "
        "first draft of a blog post. Use headings, short paragraphs, and a "
        "voice that matches the requested tone. Do not include the outline "
        "itself in the output, just the written post."
    ))
    human = HumanMessage(content=(
        f"Topic: {state['topic']}\n"
        f"Tone: {state['tone']}\n"
        f"Target length: {state['length']}\n\n"
        f"Outline to follow:\n{state['outline']}\n\n"
        "Write the full draft now."
    ))
    response = llm.invoke([system, human])
    return {"draft": response.content}


def polish_post(state: BlogState) -> dict:
    llm = _get_llm(state, temperature=0.4)
    system = SystemMessage(content=(
        "You are a meticulous editor. Improve clarity, flow, and grammar "
        "of the draft below. Keep the structure and voice. Return the "
        "final, publish-ready blog post in Markdown, with a single H1 "
        "title at the top."
    ))
    human = HumanMessage(content=(
        f"Draft:\n{state['draft']}\n\n"
        "Return only the final, polished Markdown post."
    ))
    response = llm.invoke([system, human])
    return {"final_post": response.content}


# ---------------------------------------------------------------------------
# 4. Build the graph: outline -> draft -> polish -> END
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(BlogState)

    graph.add_node("outline", create_outline)
    graph.add_node("draft", write_draft)
    graph.add_node("polish", polish_post)

    graph.set_entry_point("outline")
    graph.add_edge("outline", "draft")
    graph.add_edge("draft", "polish")
    graph.add_edge("polish", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 5. Convenience function the Streamlit app calls
# ---------------------------------------------------------------------------
def run_blog_agent(topic: str, tone: str, length: str, api_key: str,
                    model: str = "llama-3.3-70b-versatile") -> BlogState:
    """Runs the full outline -> draft -> polish pipeline and returns the
    final state dict (which includes outline, draft, and final_post)."""
    app = build_graph()
    initial_state: BlogState = {
        "topic": topic,
        "tone": tone,
        "length": length,
        "outline": "",
        "draft": "",
        "final_post": "",
        "model": model,
        "api_key": api_key,
    }
    return app.invoke(initial_state)