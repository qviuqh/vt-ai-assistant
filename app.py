from htbuilder.units import rem
from htbuilder import div, styles
from src import agent_workflow
from src.utils import generate_session_id

import datetime
import time

import streamlit as st

st.set_page_config(page_title="Streamlit AI assistant")

MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=1)
PROCESSING_TIMEOUT_SECONDS = 15

SUGGESTIONS = {
    ":blue[:material/local_library:] What is Streamlit?": (
        "What is Streamlit, what is it great at, and what can I do with it?"
    ),
    ":green[:material/database:] Help me understand session state": (
        "Help me understand session state. What is it for? "
        "What are gotchas? What are alternatives?"
    ),
    ":orange[:material/multiline_chart:] How do I make an interactive chart?": (
        "How do I make a chart where, when I click, another chart updates? "
        "Show me examples with Altair or Plotly."
    ),
    ":violet[:material/apparel:] How do I customize my app?": (
        "How do I customize my app? What does Streamlit offer? No hacks please."
    ),
    ":red[:material/deployed_code:] Deploying an app at work": (
        "How do I deploy an app at work? Give me easy and performant options."
    ),
}


def ensure_session_state():
    defaults = {
        "is_processing": False,
        "processing_started_at": None,
        "messages": [],
        "agent_connection_cache": {},
        "_last_checked_agent_id": None,
        "session_id": generate_session_id(),
        "prev_question_timestamp": datetime.datetime.fromtimestamp(0),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_conversation():
    st.session_state.messages = []
    st.session_state.initial_question = None
    st.session_state.selected_suggestion = None
    st.session_state.session_id = generate_session_id()


def history_to_text(chat_history):
    """Converts chat history into a string."""
    return "\n".join(f"[{h['role']}]: {h['content']}" for h in chat_history)


def build_question_prompt(question):
    """Builds a detailed prompt for the LLM based on the user's question and chat history."""
    selected_agent = st.session_state.get("selected_agent_id", "n8n")
    return (
        f"Session ID: {st.session_state.session_id}\n"
        f"Agent Workflow: {selected_agent}\n"
        f"Question: {question}\n\n"
        f"Chat History: {history_to_text(st.session_state.messages)}"
    )


def get_response(prompt):
    """Sends the prompt to the LLM and returns the response."""
    selected_agent = st.session_state.get("selected_agent_id", "n8n")
    return (
        f"[{selected_agent}] Hello World! This is a placeholder response. "
        "Replace this function with your LLM/agent call to get real responses."
    )


def send_telemetry(**kwargs):
    """Records some telemetry about questions being asked."""
    # TODO: Implement this.
    pass


def get_agent_status_snapshot(agent_id, force_refresh=False):
    if not agent_id:
        return None

    cache = st.session_state.agent_connection_cache
    if not force_refresh and agent_id in cache:
        return cache[agent_id]

    agent_client = None
    try:
        agent_client = agent_workflow.create_agent(agent_id)
        connection_info = agent_client.get_connection_info()
        health_info = agent_client.health_check()
    except Exception as exc:
        connection_info = {"platform": agent_id, "endpoint": None, "has_api_key": False}
        health_info = {
            "ok": False,
            "configured": False,
            "platform": agent_id,
            "url": None,
            "status_code": None,
            "message": str(exc),
        }
    finally:
        if agent_client is not None:
            agent_client.close()

    snapshot = {"connection": connection_info, "health": health_info}
    cache[agent_id] = snapshot
    st.session_state.agent_connection_cache = cache
    return snapshot


def render_sidebar():
    agent_options = agent_workflow.list_available_agents()
    agent_ids = [item["id"] for item in agent_options]
    agent_labels = {item["id"]: item["label"] for item in agent_options}

    if agent_ids:
        current_selected = st.session_state.get("selected_agent_id")
        if current_selected not in agent_ids:
            st.session_state.selected_agent_id = agent_ids[0]
    else:
        st.session_state.selected_agent_id = None

    message_count_placeholder = None

    with st.sidebar:
        st.header("Agent Workflow")

        refresh_connection = False
        if agent_ids:
            selected_agent_id = st.selectbox(
                "Select agent",
                options=agent_ids,
                format_func=lambda agent_id: agent_labels.get(agent_id, agent_id),
                key="selected_agent_id",
            )
            refresh_connection = st.button("Checking connection", use_container_width=True)
        else:
            selected_agent_id = None
            st.warning("Can not find any available agents. Please add agent configurations to proceed.")

        st.subheader("Connection status")
        if selected_agent_id:
            agent_changed = st.session_state._last_checked_agent_id != selected_agent_id
            snapshot = get_agent_status_snapshot(
                selected_agent_id,
                force_refresh=(agent_changed or refresh_connection),
            )
            st.session_state._last_checked_agent_id = selected_agent_id

            connection_info = snapshot["connection"]
            health_info = snapshot["health"]
            env_info = agent_workflow.get_agent_env_config(selected_agent_id)

            if not health_info.get("configured", True):
                st.warning(health_info.get("message", "Agent not configured"))
            elif health_info.get("ok"):
                st.success(health_info.get("message", "Connected successfully"))
            else:
                st.error(health_info.get("message", "Failed to connect"))


            st.caption(f"Platform: {connection_info.get('platform', '-')}")
            st.text_input("Endpoint", value=connection_info.get("endpoint") or "", disabled=True)
            st.caption(
                f"HTTP status: {health_info.get('status_code') if health_info.get('status_code') is not None else 'N/A'}"
            )
            st.caption(
                f"Env URL: {env_info.get('url_env') or '-'} | API key: "
                f"{'Configured' if env_info.get('has_api_key') else 'Not configured'}"
            )
        else:
            st.info("No agent selected")

        st.divider()
        st.subheader("Conversation info")
        st.text_input("session_id", value=st.session_state.session_id, disabled=True)
        message_count_placeholder = st.empty()
        message_count_placeholder.metric("Number of messages", len(st.session_state.messages))
        st.button(
            "Reset conversation",
            icon=":material/refresh:",
            on_click=clear_conversation,
            use_container_width=True,
        )

    return {"message_count_placeholder": message_count_placeholder}


@st.dialog("Legal disclaimer")
def show_disclaimer_dialog():
    st.caption("""
            
        """)


ensure_session_state()

if st.session_state.is_processing and st.session_state.processing_started_at:
    elapsed = datetime.datetime.now() - st.session_state.processing_started_at
    if elapsed.total_seconds() > PROCESSING_TIMEOUT_SECONDS:
        st.session_state.is_processing = False
        st.session_state.processing_started_at = None

sidebar_slots = render_sidebar()

# Draw the UI.
st.html(div(style=styles(font_size=rem(5), line_height=1))[""])

title_row = st.container(
    horizontal=True,
    vertical_alignment="bottom",
)

with title_row:
    st.title(
        "AI assistant",
        anchor=False,
        width="stretch",
    )

user_just_asked_initial_question = (
    "initial_question" in st.session_state and st.session_state.initial_question
)

user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)

user_first_interaction = user_just_asked_initial_question or user_just_clicked_suggestion
has_message_history = len(st.session_state.messages) > 0

# Show a different UI when the user hasn't asked a question yet.
if not user_first_interaction and not has_message_history:
    with st.container():
        st.chat_input(
            "Ask a question...",
            key="initial_question",
            disabled=st.session_state.is_processing,
        )

        st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion",
        )

    st.button(
        "&nbsp;:small[:gray[:material/balance: Legal disclaimer]]",
        type="tertiary",
        on_click=show_disclaimer_dialog,
    )

    st.stop()

# Show chat input at the bottom when a question has been asked.
user_message = st.chat_input(
    "Ask a follow-up...",
    disabled=st.session_state.is_processing,
)

if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]

# Display chat messages from history as speech bubbles.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.container()  # Fix ghost message bug.

        st.markdown(message["content"])

if user_message:
    st.session_state.is_processing = True
    st.session_state.processing_started_at = datetime.datetime.now()
    try:
        # Streamlit's Markdown engine interprets "$" as LaTeX code.
        user_message = user_message.replace("$", r"\$")

        with st.chat_message("user"):
            st.text(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Waiting..."):
                question_timestamp = datetime.datetime.now()
                time_diff = question_timestamp - st.session_state.prev_question_timestamp
                st.session_state.prev_question_timestamp = question_timestamp

                if time_diff < MIN_TIME_BETWEEN_REQUESTS:
                    time.sleep((MIN_TIME_BETWEEN_REQUESTS - time_diff).total_seconds())

                user_message = user_message.replace("'", "")

            with st.spinner("Researching..."):
                full_prompt = build_question_prompt(user_message)

            with st.spinner("Thinking..."):
                response_gen = get_response(full_prompt)

            with st.container():
                response = response_gen
                st.write(response)

                st.session_state.messages.append({"role": "user", "content": user_message})
                st.session_state.messages.append({"role": "assistant", "content": response})

                if sidebar_slots.get("message_count_placeholder") is not None:
                    sidebar_slots["message_count_placeholder"].metric(
                        "So luot tin nhan", len(st.session_state.messages)
                    )

                send_telemetry(
                    question=user_message,
                    response=response,
                    session_id=st.session_state.session_id,
                    agent_id=st.session_state.get("selected_agent_id"),
                )
    finally:
        st.session_state.is_processing = False
        st.session_state.processing_started_at = None
