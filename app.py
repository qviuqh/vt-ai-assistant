from htbuilder.units import rem
from htbuilder import div, styles
from src import agent_workflow
from src.utils import generate_session_id

import datetime
import time

import streamlit as st
import os
import json
import hashlib
import logging
import urllib.request
import urllib.error
from typing import Optional

st.set_page_config(page_title="AI Agent", layout="wide")

MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=1)
PROCESSING_TIMEOUT_SECONDS = 300

SUGGESTIONS = {
    ":blue[:material/local_library:] Lịch trình hôm nay": (
        "Lịch trình hôm nay của tôi như thế nào?"
    ),
    ":green[:material/database:] Help me understand session state": (
        "Help me understand session state. What is it for? "
        "What are gotchas? What are alternatives?"
    ),
    ":orange[:material/multiline_chart:] Tạo lịch họp sáng nay": (
        "Tạo một lịch họp cho buổi sáng nay"
    ),
    ":violet[:material/apparel:] Lịch trình tuần này": (
        "Lịch trình tuần này của tôi như thế nào?"
    ),
    ":red[:material/deployed_code:] Thông tin chung về Viettel": (
        "Cho tôi biết thông tin chung về tập đoàn Viettel?"
        "Viettel được thành lập khi nào?"
        "Viettel hoạt động trên bao nhiêu Quốc gia?"
    ),
}


def ensure_session_state():
    defaults = {
        "is_processing": False,
        "processing_started_at": None,
        "pending_user_message": None,
        "messages": [],
        "agent_connection_cache": {},
        "_last_checked_agent_id": None,
        "_last_conversation_agent_id": None,
        "session_id": generate_session_id(),
        "prev_question_timestamp": datetime.datetime.fromtimestamp(0),
        "selected_agent_id": None,
        "third_party_access_token": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_conversation():
    st.session_state.messages = []
    st.session_state.initial_question = None
    st.session_state.selected_suggestion = None
    st.session_state.pending_user_message = None
    st.session_state.is_processing = False
    st.session_state.processing_started_at = None
    st.session_state.session_id = generate_session_id()


def handle_agent_platform_change():
    selected_agent_id = st.session_state.get("selected_agent_id")
    previous_agent_id = st.session_state.get("_last_conversation_agent_id")

    if previous_agent_id and previous_agent_id != selected_agent_id:
        clear_conversation()

    st.session_state._last_conversation_agent_id = selected_agent_id


def history_to_text(chat_history):
    return "\n".join(f"[{h['role']}]: {h['content']}" for h in chat_history)


def build_question_prompt(question):
    selected_agent = st.session_state.get("selected_agent_id", "n8n")
    return (
        f"Session ID: {st.session_state.session_id}\n"
        f"Agent Workflow: {selected_agent}\n"
        f"Question: {question}\n\n"
        f"Chat History: {history_to_text(st.session_state.messages)}"
    )


def get_response_stream(question: str):
    """Returns a generator that yields text chunks from the selected agent."""
    agent_id = st.session_state.get("selected_agent_id")
    if not agent_id:
        yield "⚠️ No agent selected."
        return

    agent_client = None
    try:
        agent_client = agent_workflow.create_agent(
            agent_id,
            access_token=st.session_state.get("third_party_access_token", ""),
        )
        yield from agent_client.stream_response(question, st.session_state.session_id)
    except Exception as exc:
        yield f"⚠️ Error: {exc}"
    finally:
        if agent_client is not None:
            agent_client.close()


def send_telemetry(**kwargs):
    """Send lightweight telemetry for analytics/debugging.

    Behavior:
    - Disabled when env var `TELEMETRY_DISABLED` is set to true/1/yes.
    - If `TELEMETRY_ENDPOINT` is set, POSTs JSON to that URL.
      Optional API key may be passed via `TELEMETRY_API_KEY` (Bearer).
    - If no endpoint is configured, appends events to `telemetry.log`
      next to `app.py`.

    To minimize accidental PII leakage we include only short snippets
    and SHA256 hashes of `question` and `response` instead of full text.
    """
    try:
        if os.getenv("TELEMETRY_DISABLED", "").lower() in ("1", "true", "yes"):
            return

        # Extract common fields
        question = kwargs.get("question")
        response = kwargs.get("response")
        session_id = kwargs.get("session_id")
        agent_id = kwargs.get("agent_id")

        def _hash(s: Optional[str]) -> Optional[str]:
            if s is None:
                return None
            return hashlib.sha256(s.encode("utf-8")).hexdigest()

        payload = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "agent_id": agent_id,
            "question_hash": _hash(question),
            "response_hash": _hash(response),
            "question_snippet": (question[:200] + "…") if question and len(question) > 200 else question,
            "response_snippet": (response[:200] + "…") if response and len(response) > 200 else response,
            "question_length": len(question) if question else 0,
            "response_length": len(response) if response else 0,
        }

        # copy any other non-sensitive kwargs
        for k, v in kwargs.items():
            if k in ("question", "response", "session_id", "agent_id"):
                continue
            payload[k] = v

        endpoint = os.getenv("TELEMETRY_ENDPOINT")
        api_key = os.getenv("TELEMETRY_API_KEY")

        if endpoint:
            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    # read response to complete request; ignore content
                    _ = resp.read()
            except Exception as exc:  # network/timeout errors should not break app
                logging.getLogger(__name__).exception("Failed to send telemetry: %s", exc)
        else:
            # Fallback: append to local log file next to app.py
            try:
                log_path = os.path.join(os.path.dirname(__file__), "telemetry.log")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception as exc:
                logging.getLogger(__name__).exception("Failed to write telemetry log: %s", exc)
    except Exception as exc:
        logging.getLogger(__name__).exception("Unexpected error in send_telemetry: %s", exc)


def get_agent_status_snapshot(agent_id, force_refresh=False):
    if not agent_id:
        return None

    cache = st.session_state.agent_connection_cache
    if not force_refresh and agent_id in cache:
        return cache[agent_id]

    agent_client = None
    try:
        agent_client = agent_workflow.create_agent(
            agent_id,
            access_token=st.session_state.get("third_party_access_token", ""),
        )
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

    if agent_ids and st.session_state.get("selected_agent_id") not in agent_ids:
        st.session_state.selected_agent_id = agent_ids[0]

    if (
        st.session_state.get("_last_conversation_agent_id") is None
        and st.session_state.get("selected_agent_id") in agent_ids
    ):
        st.session_state._last_conversation_agent_id = st.session_state.selected_agent_id

    message_count_placeholder = None

    with st.sidebar:
        st.header("Agent Workflow")

        if not agent_ids:
            st.warning("No agents available.")
            return {"message_count_placeholder": None}

        selected_agent_id = st.selectbox(
            "Platform",
            options=agent_ids,
            format_func=lambda aid: agent_labels.get(aid, aid),
            key="selected_agent_id",
            on_change=handle_agent_platform_change,
        )

        refresh_connection = st.button(
            "🔄 Refresh",
            use_container_width=True,
            help="Re-check connection to the selected platform",
        )

        # --- Connection status ---
        st.subheader("Connection Status")

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

            # Status badge
            if not health_info.get("configured", True):
                st.warning(f"{health_info.get('message', 'Not configured')}")
            elif health_info.get("ok"):
                st.success(f"{health_info.get('message', 'Connected')}")
            else:
                st.error(f"{health_info.get('message', 'Failed to connect')}")

            endpoint = connection_info.get("endpoint") or ""
            st.text_input("Endpoint", value=endpoint, disabled=True)

            col_a, col_b = st.columns(2)
            with col_a:
                status_code = health_info.get("status_code")
                st.metric("HTTP Status", status_code if status_code else "—")
            with col_b:
                api_key_status = "Set" if env_info.get("has_api_key") else "Missing"
                st.metric("API Key", api_key_status)

            env_var = env_info.get("url_env", "")
            if env_var:
                st.caption(f"URL env var: `{env_var}`")
        else:
            st.info("No agent selected")

        st.divider()

        # --- Runtime headers ---
        st.subheader("Runtime Headers")
        st.text_input(
            "access_token",
            key="third_party_access_token",
            type="password",
            help=(
                "Optional token sent to the selected platform in request headers "
                "as `access_token` so downstream tools (e.g. n8n) can use it."
            ),
            placeholder="Paste third-party access token",
        )
        st.caption("Outgoing requests include header: `access_token` (when provided).")

        st.divider()

        # --- Conversation info ---
        st.subheader("Conversation")

        st.text_input(
            "Session ID",
            value=st.session_state.session_id,
            disabled=True,
            help="Unique identifier for the current conversation session",
        )

        message_count_placeholder = st.empty()
        _update_message_metric(message_count_placeholder)

        st.button(
            "Reset Conversation",
            on_click=clear_conversation,
            use_container_width=True,
            type="secondary",
        )

        st.divider()

        # --- Platform quick guide ---
        with st.expander("Platform setup guide", expanded=False):
            _render_setup_guide(selected_agent_id)

    return {"message_count_placeholder": message_count_placeholder}


def _update_message_metric(placeholder):
    count = len(st.session_state.messages)
    turns = count // 2
    placeholder.metric(
        "Messages",
        count,
        help=f"{turns} conversation turn{'s' if turns != 1 else ''}",
    )


def _platform_docs_url(agent_id: str) -> str:
    urls = {
        "n8n": "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/",
        "dify": "https://docs.dify.ai/guides/application-publishing/developing-with-apis",
        "langflow": "https://docs.langflow.org/api-reference/overview",
        "flowise": "https://docs.flowiseai.com/api-reference/prediction",
    }
    return urls.get(agent_id, "https://google.com")


def _render_setup_guide(agent_id: str):
    guides = {
        "n8n": """
**Required env vars:**
- `N8N_WEBHOOK_URL` – your webhook URL
- `N8N_API_KEY` *(optional)* – bearer token

**Workflow tips:**
- Use a Webhook node as trigger
- Return JSON with an `output` field
- Enable streaming via SSE for better UX
""",
        "dify": """
**Required env vars:**
- `DIFY_API_KEY` – your app's API key
- `DIFY_API_URL` *(optional)* – default: api.dify.ai/v1
- `DIFY_USER` *(optional)* – user identifier

**Tips:**
- Use a "Chat" type application
- Streaming is enabled automatically
""",
        "langflow": """
**Required env vars:**
- `LANGFLOW_FLOW_ID` – UUID from the flow URL
- `LANGFLOW_API_URL` *(optional)* – default: localhost:7860
- `LANGFLOW_API_KEY` *(optional)*

**Tips:**
- Expose your flow via API in the UI
- Add a Chat Input + Chat Output component
""",
        "flowise": """
**Required env vars:**
- `FLOWISE_CHATFLOW_ID` – chatflow UUID
- `FLOWISE_API_URL` *(optional)* – default: localhost:3000
- `FLOWISE_API_KEY` *(optional)*

**Tips:**
- Enable streaming in chatflow settings
- Use the Upsert API to inject documents
""",
    }
    st.markdown(guides.get(agent_id, "No guide available for this platform."))


@st.dialog("Legal disclaimer")
def show_disclaimer_dialog():
    st.caption("This is a demo application. Use responsibly.")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

ensure_session_state()

# Timeout guard
if st.session_state.is_processing and st.session_state.processing_started_at:
    elapsed = datetime.datetime.now() - st.session_state.processing_started_at
    if elapsed.total_seconds() > PROCESSING_TIMEOUT_SECONDS:
        st.session_state.is_processing = False
        st.session_state.processing_started_at = None
        st.session_state.pending_user_message = None

sidebar_slots = render_sidebar()

# Header
st.html(div(style=styles(font_size=rem(5), line_height=1))[""])
st.title("AI Agent Playground", anchor=False)

# State flags
user_just_asked_initial_question = (
    "initial_question" in st.session_state and st.session_state.initial_question
)
user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)
user_first_interaction = user_just_asked_initial_question or user_just_clicked_suggestion
has_message_history = len(st.session_state.messages) > 0
has_pending_user_message = bool(st.session_state.get("pending_user_message"))
is_chat_input_blocked = st.session_state.is_processing or has_pending_user_message

# Landing page (no messages yet)
if not user_first_interaction and not has_message_history and not has_pending_user_message:
    with st.container():
        st.chat_input(
            "Ask a question...",
            key="initial_question",
            disabled=is_chat_input_blocked,
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

# Follow-up chat input
incoming_user_message = st.chat_input(
    "Ask a follow-up...",
    disabled=is_chat_input_blocked,
)

if (
    not incoming_user_message
    and not is_chat_input_blocked
):
    if user_just_asked_initial_question:
        incoming_user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        incoming_user_message = SUGGESTIONS[st.session_state.selected_suggestion]

if incoming_user_message and not is_chat_input_blocked:
    st.session_state.pending_user_message = incoming_user_message
    st.session_state.is_processing = True
    st.session_state.processing_started_at = datetime.datetime.now()
    st.session_state.initial_question = None
    st.session_state.selected_suggestion = None
    st.rerun()

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.container()  # fix ghost message bug
        st.markdown(message["content"])

# Handle queued user message
pending_user_message = st.session_state.get("pending_user_message")
if pending_user_message:
    try:
        user_message = pending_user_message.replace("$", r"\$")

        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Connecting to agent..."):
                question_timestamp = datetime.datetime.now()
                time_diff = question_timestamp - st.session_state.prev_question_timestamp
                st.session_state.prev_question_timestamp = question_timestamp

                if time_diff < MIN_TIME_BETWEEN_REQUESTS:
                    time.sleep((MIN_TIME_BETWEEN_REQUESTS - time_diff).total_seconds())

            # Stream the response
            response_chunks = []

            def _stream():
                for chunk in get_response_stream(user_message):
                    response_chunks.append(chunk)
                    yield chunk

            response_text = st.write_stream(_stream())

            # Fallback if write_stream returned None
            if not response_text:
                response_text = "".join(response_chunks)

            st.session_state.messages.append({"role": "user", "content": user_message})
            st.session_state.messages.append({"role": "assistant", "content": response_text})

            if sidebar_slots.get("message_count_placeholder") is not None:
                _update_message_metric(sidebar_slots["message_count_placeholder"])

            send_telemetry(
                question=user_message,
                response=response_text,
                session_id=st.session_state.session_id,
                agent_id=st.session_state.get("selected_agent_id"),
            )
    finally:
        st.session_state.is_processing = False
        st.session_state.processing_started_at = None
        st.session_state.pending_user_message = None

    st.rerun()
