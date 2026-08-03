"""
app.py
------
Streamlit frontend for the LangGraph + Groq blog-writing agent defined in
agent.py.

Run with:
    streamlit run app.py
"""

import streamlit as st
from agent import build_graph

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Blog Writer Agent",
    page_icon="📝",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main > div { padding-top: 2rem; }
        .stApp { background-color: #faf9f6; }
        h1 { color: #1a1a1a; font-weight: 800; }
        .agent-card {
            background: #ffffff;
            border: 1px solid #e8e6e1;
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }
        .step-badge {
            display: inline-block;
            background: #d97757;
            color: white;
            border-radius: 999px;
            padding: 0.15rem 0.7rem;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📝 Blog Writer Agent")
st.caption("LangChain + LangGraph agent, powered by Groq — outline → draft → polish")

# ---------------------------------------------------------------------------
# Sidebar: API key + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com/keys. "
             "It is only used for this session and never stored.",
    )

    model = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ],
        index=0,
    )

    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. `outline` node plans the post\n"
        "2. `draft` node writes it in full\n"
        "3. `polish` node edits it for clarity\n\n"
        "Each step is a node in a LangGraph graph, and each LLM call goes "
        "through Groq."
    )

# ---------------------------------------------------------------------------
# Main form
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    topic = st.text_input("Blog topic", placeholder="e.g. Why async programming matters for beginners")

with col2:
    tone = st.selectbox("Tone", ["Friendly", "Formal", "Witty", "Technical", "Persuasive"])

length = st.select_slider("Target length", options=["Short", "Medium", "Long"], value="Medium")

generate = st.button("✨ Generate blog post", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------
if generate:
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar first.")
    elif not topic:
        st.error("Please enter a blog topic.")
    else:
        app = build_graph()
        state = {
            "topic": topic,
            "tone": tone,
            "length": length,
            "outline": "",
            "draft": "",
            "final_post": "",
            "model": model,
            "api_key": groq_api_key,
        }

        outline_box = st.empty()
        draft_box = st.empty()
        final_box = st.empty()

        try:
            with st.status("Running the agent...", expanded=True) as status:
                st.write("🧭 Planning the outline...")
                for event in app.stream(state):
                    node_name = list(event.keys())[0]
                    node_output = event[node_name]
                    state.update(node_output)

                    if node_name == "outline":
                        status.write("✅ Outline ready. Writing the full draft...")
                        with outline_box.container():
                            st.markdown('<span class="step-badge">STEP 1 · OUTLINE</span>', unsafe_allow_html=True)
                            with st.expander("View outline", expanded=False):
                                st.markdown(state["outline"])

                    elif node_name == "draft":
                        status.write("✅ Draft ready. Polishing for the final version...")
                        with draft_box.container():
                            st.markdown('<span class="step-badge">STEP 2 · DRAFT</span>', unsafe_allow_html=True)
                            with st.expander("View draft", expanded=False):
                                st.markdown(state["draft"])

                    elif node_name == "polish":
                        status.update(label="Done!", state="complete", expanded=False)

            st.success("Your blog post is ready 🎉")
            st.markdown('<span class="step-badge">FINAL POST</span>', unsafe_allow_html=True)
            with st.container():
                st.markdown(state["final_post"])

            st.download_button(
                "⬇️ Download as Markdown",
                data=state["final_post"],
                file_name=f"{topic.strip().replace(' ', '_').lower()}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.info("Double-check your Groq API key and selected model, then try again.")
else:
    st.markdown(
        """
        <div class="agent-card">
        👋 Enter your Groq API key in the sidebar, pick a topic, tone, and
        length, then hit <b>Generate blog post</b>. You'll see the agent
        move through its three steps — outline, draft, and polish — live.
        </div>
        """,
        unsafe_allow_html=True,
    )