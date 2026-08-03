# Blog Writer Agent (LangChain + LangGraph + Groq + Streamlit)

A minimal agent that writes a blog post in three steps — **outline → draft →
polish** — using a tiny [LangGraph](https://langchain-ai.github.io/langgraph/)
graph, [LangChain](https://python.langchain.com/)'s Groq integration, and a
[Streamlit](https://streamlit.io/) frontend.

## Files

- `agent.py` — the agent's "brain": LangGraph state, node functions, and graph.
- `app.py` — the Streamlit UI. Takes the user's Groq API key at runtime (never hard-coded).
- `requirements.txt` — dependencies.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Usage

1. Get a free Groq API key at https://console.groq.com/keys
2. Paste it into the sidebar (it's only used for your session, never stored or logged).
3. Enter a topic, pick a tone and length.
4. Click **Generate blog post** and watch the agent move through its steps live.
5. Download the final post as Markdown.

## How the graph works

```
        ┌──────────┐      ┌───────┐      ┌────────┐
 start →│ outline  │ ───► │ draft │ ───► │ polish │ ───► END
        └──────────┘      └───────┘      └────────┘
```

Each node is a plain Python function that:
1. Reads relevant fields from the shared `BlogState` (a `TypedDict`)
2. Calls `ChatGroq` with a system + human message
3. Returns a partial state update (e.g. `{"outline": "..."}`), which LangGraph merges into the running state

This is intentionally simple — no tools, no branching, no loops — so it's easy
to read end-to-end and extend (e.g. add a "research" node with web search, or
a "human review" node that pauses for edits).

<img width="1919" height="913" alt="image" src="https://github.com/user-attachments/assets/a965e181-9763-4da0-bdf0-e82612931f96" />
<img width="1915" height="907" alt="image" src="https://github.com/user-attachments/assets/430312ef-ea08-4eac-b8ad-2dbb74d52587" />
<img width="1916" height="909" alt="image" src="https://github.com/user-attachments/assets/496a018d-abd2-4a3a-aaa3-38dce6e9a67d" />
<img width="1917" height="906" alt="image" src="https://github.com/user-attachments/assets/0cc9a1c7-9a39-4ef2-adfe-b32e7bf503aa" />



