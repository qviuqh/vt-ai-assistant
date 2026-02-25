from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Generator, List
import os
import json
import logging
from pathlib import Path

import requests
import sseclient

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    _load_dotenv = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_project_env() -> None:
    """Load .env from the repository root so os.getenv() works in Streamlit/Jupyter."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    if _load_dotenv is not None:
        _load_dotenv(dotenv_path=env_path, override=False)
        logger.info("Loaded environment variables from %s", env_path)
        return

    # Basic fallback parser when python-dotenv is not installed.
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)

    logger.warning(
        "python-dotenv is not installed. Loaded .env with basic parser from %s",
        env_path,
    )


_load_project_env()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AgentWorkflow(ABC):
    """Base class for AI Agent workflow integrations."""

    def __init__(self, name: str, access_token: str = ""):
        self.name = name
        self.access_token = (access_token or "").strip()
        self.session = requests.Session()
        logger.info("Initialized %s agent", self.name)

    @abstractmethod
    def send_request(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def process_response(self, response: Any, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_connection_info(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def stream_response(self, query: str, session_id: str) -> Generator[str, None, None]:
        """
        Unified streaming interface used by app.py.
        Yields plain-text chunks that can be piped directly into st.write_stream().
        """
        pass

    def close(self):
        self.session.close()
        logger.info("Closed %s agent session", self.name)

    def _with_access_token_header(
        self, headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        merged_headers = dict(headers or {})
        if self.access_token:
            merged_headers["access_token"] = self.access_token
        return merged_headers

    def _missing_config_result(self, url: Optional[str], missing: List[str]) -> Dict[str, Any]:
        fields = ", ".join(missing)
        return {
            "ok": False,
            "configured": False,
            "platform": self.name,
            "url": url,
            "status_code": None,
            "message": f"Missing configuration: {fields}",
        }

    def _health_check_request(
        self,
        url: Optional[str],
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 5,
        method: str = "GET",
    ) -> Dict[str, Any]:
        """Lightweight connectivity probe. Any HTTP < 500 is treated as reachable."""
        if not url:
            return self._missing_config_result(url, ["url"])

        try:
            response = self.session.request(
                method=method, url=url, headers=headers or {}, timeout=timeout
            )
            status_code = response.status_code
            ok = status_code < 500

            if 200 <= status_code < 300:
                message = "Connected successfully"
            elif status_code in (401, 403):
                message = f"Reachable but authentication failed (HTTP {status_code})"
            elif status_code in (404, 405):
                message = f"Reachable (HTTP {status_code})"
            else:
                message = f"HTTP {status_code}"

            return {
                "ok": ok,
                "configured": True,
                "platform": self.name,
                "url": url,
                "status_code": status_code,
                "message": message,
            }
        except requests.exceptions.ConnectionError as exc:
            return {
                "ok": False,
                "configured": True,
                "platform": self.name,
                "url": url,
                "status_code": None,
                "message": f"Connection refused – is the server running? ({exc})",
            }
        except requests.exceptions.Timeout:
            return {
                "ok": False,
                "configured": True,
                "platform": self.name,
                "url": url,
                "status_code": None,
                "message": "Connection timed out",
            }
        except requests.exceptions.RequestException as exc:
            logger.error("Health check failed for %s: %s", self.name, str(exc))
            return {
                "ok": False,
                "configured": True,
                "platform": self.name,
                "url": url,
                "status_code": None,
                "message": str(exc),
            }


# ---------------------------------------------------------------------------
# n8n
# ---------------------------------------------------------------------------

class N8nAgent(AgentWorkflow):
    """
    Connects to an n8n Webhook-triggered workflow.

    Environment variables:
        N8N_WEBHOOK_URL   – full webhook URL, e.g. http://localhost:5678/webhook/my-bot
        N8N_API_KEY       – optional bearer token added as Authorization header
    """

    PLATFORM = "n8n"

    def __init__(self, access_token: str = ""):
        super().__init__(self.PLATFORM, access_token=access_token)
        self.webhook_url = os.getenv("N8N_WEBHOOK_URL", "").rstrip("/")
        self.api_key = os.getenv("N8N_API_KEY", "")

    # -- internal helpers ---------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return self._with_access_token_header(h)

    # -- AgentWorkflow interface --------------------------------------------

    def get_connection_info(self) -> Dict[str, Any]:
        return {
            "platform": self.PLATFORM,
            "endpoint": self.webhook_url or None,
            "has_api_key": bool(self.api_key),
        }

    def health_check(self) -> Dict[str, Any]:
        if not self.webhook_url:
            return self._missing_config_result(None, ["N8N_WEBHOOK_URL"])
        return self._health_check_request(self.webhook_url, headers=self._headers(), method="GET")

    def send_request(self, query: str, session_id: str, chat_history: list = None) -> requests.Response:
        payload = {
            "query": query,
            "sessionId": session_id,
            "chatHistory": chat_history or [],
        }
        response = self.session.post(
            self.webhook_url,
            json=payload,
            headers=self._headers(),
            timeout=60,
            stream=True,
        )
        response.raise_for_status()
        return response

    def process_response(self, response: requests.Response, **kwargs) -> str:
        """
        n8n webhooks can return:
          • plain text
          • {"output": "..."}
          • {"message": "..."}
          • {"text": "..."}
        """
        content_type = response.headers.get("Content-Type", "")
        text = response.text.strip()

        if "application/json" in content_type:
            try:
                data = response.json()
                if isinstance(data, list) and data:
                    data = data[0]
                return (
                    data.get("output")
                    or data.get("message")
                    or data.get("text")
                    or data.get("response")
                    or json.dumps(data)
                )
            except (ValueError, AttributeError):
                return text
        return text

    def stream_response(self, query: str, session_id: str) -> Generator[str, None, None]:
        if not self.webhook_url:
            yield "⚠️ N8N_WEBHOOK_URL is not configured."
            return
        try:
            resp = self.send_request(query, session_id)
            content_type = resp.headers.get("Content-Type", "")

            if "text/event-stream" in content_type:
                client = sseclient.SSEClient(resp)
                for event in client.events():
                    if event.data and event.data != "[DONE]":
                        try:
                            chunk = json.loads(event.data)
                            yield (
                                chunk.get("output")
                                or chunk.get("text")
                                or chunk.get("message")
                                or ""
                            )
                        except ValueError:
                            yield event.data
            else:
                yield self.process_response(resp)
        except requests.exceptions.RequestException as exc:
            yield f"⚠️ Request failed: {exc}"


# ---------------------------------------------------------------------------
# Dify
# ---------------------------------------------------------------------------

class DifyAgent(AgentWorkflow):
    """
    Connects to Dify's Chat-messages API (blocking or streaming).

    Environment variables:
        DIFY_API_URL     – base URL, default https://api.dify.ai/v1
        DIFY_API_KEY     – app API key (required)
        DIFY_USER        – user identifier sent to Dify (default "streamlit-user")
    """

    PLATFORM = "dify"
    DEFAULT_BASE_URL = "https://api.dify.ai/v1"

    def __init__(self, access_token: str = ""):
        super().__init__(self.PLATFORM, access_token=access_token)
        self.base_url = os.getenv("DIFY_API_URL", self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key = os.getenv("DIFY_API_KEY", "")
        self.user = os.getenv("DIFY_USER", "streamlit-user")

    def _headers(self) -> Dict[str, str]:
        return self._with_access_token_header({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def get_connection_info(self) -> Dict[str, Any]:
        return {
            "platform": self.PLATFORM,
            "endpoint": self.base_url,
            "has_api_key": bool(self.api_key),
        }

    def health_check(self) -> Dict[str, Any]:
        missing = []
        if not self.api_key:
            missing.append("DIFY_API_KEY")
        if missing:
            return self._missing_config_result(self.base_url, missing)
        # Hit the parameters endpoint as a lightweight probe
        url = f"{self.base_url}/parameters"
        return self._health_check_request(url, headers=self._headers())

    def send_request(
        self,
        query: str,
        session_id: str,
        conversation_id: str = "",
        stream: bool = True,
    ) -> requests.Response:
        payload = {
            "inputs": {},
            "query": query,
            "response_mode": "streaming" if stream else "blocking",
            "conversation_id": conversation_id,
            "user": self.user,
        }
        response = self.session.post(
            f"{self.base_url}/chat-messages",
            json=payload,
            headers=self._headers(),
            timeout=60,
            stream=stream,
        )
        response.raise_for_status()
        return response

    def process_response(self, response: requests.Response, **kwargs) -> str:
        data = response.json()
        return data.get("answer") or data.get("message") or json.dumps(data)

    def stream_response(self, query: str, session_id: str) -> Generator[str, None, None]:
        if not self.api_key:
            yield "⚠️ DIFY_API_KEY is not configured."
            return
        try:
            resp = self.send_request(query, session_id, stream=True)
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]" or not line:
                    continue
                try:
                    chunk = json.loads(line)
                    event = chunk.get("event", "")
                    if event == "message":
                        yield chunk.get("answer", "")
                    elif event == "agent_message":
                        yield chunk.get("answer", "")
                    elif event == "message_end":
                        break
                    elif event == "error":
                        yield f"\n⚠️ {chunk.get('message', 'Unknown error')}"
                        break
                except ValueError:
                    yield line
        except requests.exceptions.RequestException as exc:
            yield f"⚠️ Request failed: {exc}"


# ---------------------------------------------------------------------------
# LangFlow
# ---------------------------------------------------------------------------

class LangFlowAgent(AgentWorkflow):
    """
    Connects to a LangFlow flow via its /api/v1/run/{flow_id} endpoint.

    Environment variables:
        LANGFLOW_API_URL  – base URL, e.g. http://localhost:7860
        LANGFLOW_FLOW_ID  – UUID of the flow to run (required)
        LANGFLOW_API_KEY  – optional API key
    """

    PLATFORM = "langflow"

    def __init__(self, access_token: str = ""):
        super().__init__(self.PLATFORM, access_token=access_token)
        self.base_url = os.getenv("LANGFLOW_API_URL", "http://localhost:7860").rstrip("/")
        self.flow_id = os.getenv("LANGFLOW_FLOW_ID", "")
        self.api_key = os.getenv("LANGFLOW_API_KEY", "")

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return self._with_access_token_header(h)

    def get_connection_info(self) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/api/v1/run/{self.flow_id}" if self.flow_id else self.base_url
        return {
            "platform": self.PLATFORM,
            "endpoint": endpoint,
            "has_api_key": bool(self.api_key),
        }

    def health_check(self) -> Dict[str, Any]:
        if not self.flow_id:
            return self._missing_config_result(self.base_url, ["LANGFLOW_FLOW_ID"])
        url = f"{self.base_url}/api/v1/flows/{self.flow_id}"
        return self._health_check_request(url, headers=self._headers())

    def send_request(self, query: str, session_id: str, stream: bool = True) -> requests.Response:
        url = f"{self.base_url}/api/v1/run/{self.flow_id}"
        payload = {
            "input_value": query,
            "output_type": "chat",
            "input_type": "chat",
            "session_id": session_id,
            "stream": stream,
        }
        response = self.session.post(
            url,
            json=payload,
            headers=self._headers(),
            timeout=60,
            stream=stream,
            params={"stream": "true"} if stream else {},
        )
        response.raise_for_status()
        return response

    def process_response(self, response: requests.Response, **kwargs) -> str:
        data = response.json()
        try:
            outputs = data["outputs"][0]["outputs"][0]
            return (
                outputs.get("results", {}).get("message", {}).get("text")
                or outputs.get("messages", [{}])[0].get("message")
                or json.dumps(data)
            )
        except (KeyError, IndexError, TypeError):
            return json.dumps(data)

    def stream_response(self, query: str, session_id: str) -> Generator[str, None, None]:
        if not self.flow_id:
            yield "⚠️ LANGFLOW_FLOW_ID is not configured."
            return
        try:
            resp = self.send_request(query, session_id, stream=True)
            content_type = resp.headers.get("Content-Type", "")

            if "text/event-stream" in content_type:
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(line)
                        yield chunk.get("chunk") or chunk.get("text") or ""
                    except ValueError:
                        yield line
            else:
                yield self.process_response(resp)
        except requests.exceptions.RequestException as exc:
            yield f"⚠️ Request failed: {exc}"


# ---------------------------------------------------------------------------
# Flowise
# ---------------------------------------------------------------------------

class FlowiseAgent(AgentWorkflow):
    """
    Connects to a Flowise chatflow via its /api/v1/prediction/{chatflow_id} endpoint.

    Environment variables:
        FLOWISE_API_URL      – base URL, e.g. http://localhost:3000
        FLOWISE_CHATFLOW_ID  – chatflow UUID (required)
        FLOWISE_API_KEY      – optional bearer token
    """

    PLATFORM = "flowise"

    def __init__(self, access_token: str = ""):
        super().__init__(self.PLATFORM, access_token=access_token)
        self.base_url = os.getenv("FLOWISE_API_URL", "http://localhost:3000").rstrip("/")
        self.chatflow_id = os.getenv("FLOWISE_CHATFLOW_ID", "")
        self.api_key = os.getenv("FLOWISE_API_KEY", "")

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return self._with_access_token_header(h)

    def get_connection_info(self) -> Dict[str, Any]:
        endpoint = (
            f"{self.base_url}/api/v1/prediction/{self.chatflow_id}"
            if self.chatflow_id
            else self.base_url
        )
        return {
            "platform": self.PLATFORM,
            "endpoint": endpoint,
            "has_api_key": bool(self.api_key),
        }

    def health_check(self) -> Dict[str, Any]:
        if not self.chatflow_id:
            return self._missing_config_result(self.base_url, ["FLOWISE_CHATFLOW_ID"])
        url = f"{self.base_url}/api/v1/chatflows/{self.chatflow_id}"
        return self._health_check_request(url, headers=self._headers())

    def send_request(self, query: str, session_id: str, chat_history: list = None) -> requests.Response:
        url = f"{self.base_url}/api/v1/prediction/{self.chatflow_id}"
        payload = {
            "question": query,
            "sessionId": session_id,
            "streaming": True,
            "history": [
                {"role": m["role"], "content": m["content"]}
                for m in (chat_history or [])
            ],
        }
        response = self.session.post(
            url,
            json=payload,
            headers=self._headers(),
            timeout=60,
            stream=True,
        )
        response.raise_for_status()
        return response

    def process_response(self, response: requests.Response, **kwargs) -> str:
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            data = response.json()
            return (
                data.get("text")
                or data.get("answer")
                or data.get("output")
                or json.dumps(data)
            )
        return response.text.strip()

    def stream_response(self, query: str, session_id: str) -> Generator[str, None, None]:
        if not self.chatflow_id:
            yield "⚠️ FLOWISE_CHATFLOW_ID is not configured."
            return
        try:
            resp = self.send_request(query, session_id)
            content_type = resp.headers.get("Content-Type", "")

            if "text/event-stream" in content_type:
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(line)
                        event = chunk.get("event", "token")
                        if event in ("token", "message"):
                            yield chunk.get("data", "") or chunk.get("text", "")
                        elif event == "end":
                            break
                        elif event == "error":
                            yield f"\n⚠️ {chunk.get('data', 'Unknown error')}"
                            break
                    except ValueError:
                        yield line
            else:
                yield self.process_response(resp)
        except requests.exceptions.RequestException as exc:
            yield f"⚠️ Request failed: {exc}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "n8n": {"label": "n8n Workflow", "cls": N8nAgent},
    "dify": {"label": "Dify", "cls": DifyAgent},
    "langflow": {"label": "LangFlow", "cls": LangFlowAgent},
    "flowise": {"label": "Flowise", "cls": FlowiseAgent},
}

_ENV_KEYS: Dict[str, Dict[str, str]] = {
    "n8n":      {"url_env": "N8N_WEBHOOK_URL",   "key_env": "N8N_API_KEY"},
    "dify":     {"url_env": "DIFY_API_URL",       "key_env": "DIFY_API_KEY"},
    "langflow": {"url_env": "LANGFLOW_API_URL",   "key_env": "LANGFLOW_API_KEY"},
    "flowise":  {"url_env": "FLOWISE_API_URL",    "key_env": "FLOWISE_API_KEY"},
}


def list_available_agents() -> List[Dict[str, str]]:
    return [{"id": k, "label": v["label"]} for k, v in _AGENT_REGISTRY.items()]


def create_agent(agent_id: str, access_token: str = "") -> AgentWorkflow:
    entry = _AGENT_REGISTRY.get(agent_id)
    if not entry:
        raise ValueError(f"Unknown agent: {agent_id}")
    return entry["cls"](access_token=access_token)


def get_agent_env_config(agent_id: str) -> Dict[str, Any]:
    keys = _ENV_KEYS.get(agent_id, {})
    url_env = keys.get("url_env", "")
    key_env = keys.get("key_env", "")
    return {
        "url_env": url_env,
        "has_api_key": bool(os.getenv(key_env, "")),
    }
