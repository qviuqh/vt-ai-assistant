from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Generator, List
import os
import json
import logging
from pathlib import Path

import requests

try:
    import sseclient
except ImportError:  # pragma: no cover - optional dependency at runtime
    sseclient = None


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"
_AGENT_REGISTRY_ENV_VAR = "AGENT_REGISTRY_JSON"


def _load_dotenv_file(dotenv_path: Path = _DOTENV_PATH) -> None:
    """
    Lightweight .env loader for local development.

    It only fills variables that are not already set in the process
    environment, so deployment/runtime-provided env vars still win.
    """
    if not dotenv_path.exists():
        return

    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[len("export "):].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

            os.environ.setdefault(key, value)
    except OSError as exc:
        logger.warning("Failed to load .env file from %s: %s", dotenv_path, str(exc))


class AgentWorkflow(ABC):
    """
    Base class for AI Agent workflows
    Provides common interface for different agent platforms
    """

    def __init__(self, name: str):
        self.name = name
        self.session = requests.Session()
        logger.info(f"Initialized {self.name} agent")

    @abstractmethod
    def send_request(self, *args, **kwargs) -> Any:
        """Send request to the agent platform"""
        pass

    @abstractmethod
    def process_response(self, response: Any, *args, **kwargs) -> Any:
        """Process the response from the agent platform"""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check connectivity to the agent platform"""
        pass

    @abstractmethod
    def get_connection_info(self) -> Dict[str, Any]:
        """Return connection metadata for UI display"""
        pass

    def close(self):
        """Clean up resources"""
        self.session.close()
        logger.info(f"Closed {self.name} agent session")

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
        """
        Performs a lightweight connectivity check.

        By default, any HTTP status <500 is treated as "reachable" because some
        webhook/chat endpoints legitimately return 401/403/404/405 to GET probes.
        """
        if not url:
            return self._missing_config_result(url, ["url"])

        try:
            response = self.session.request(method=method, url=url, headers=headers or {}, timeout=timeout)
            status_code = response.status_code
            ok = status_code < 500

            if 200 <= status_code < 300:
                message = "Connected"
            elif status_code in (401, 403):
                message = f"Reachable (authentication issue, HTTP {status_code})"
            elif status_code in (404, 405):
                message = f"Reachable (endpoint/method probe returned HTTP {status_code})"
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


class N8nAgent(AgentWorkflow):
    """
    n8n Webhook Agent with streaming support
    Handles webhook requests and streaming responses
    """

    def __init__(self, webhook_url: str, api_key: Optional[str] = None, timeout: int = 30):
        super().__init__("n8n")
        self.webhook_url = webhook_url
        self.api_key = api_key
        self.timeout = timeout

        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def send_request(self, data: Dict[str, Any], stream: bool = True) -> requests.Response:
        """Send request to n8n webhook"""
        try:
            logger.info("Sending request to n8n webhook: %s", self.webhook_url)
            response = self.session.post(
                self.webhook_url,
                json=data,
                headers=self.headers,
                stream=stream,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            logger.error("Error sending request to n8n: %s", str(exc))
            raise

    def process_response(self, response: requests.Response, stream: bool = True) -> Any:
        """Process n8n webhook response"""
        if stream:
            return self._process_streaming_response(response)
        return response.json()

    def _process_streaming_response(self, response: requests.Response) -> Generator[Any, None, None]:
        """Process streaming response from n8n webhook"""
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if line.startswith("data: "):
                    line = line[6:]

                if line.strip() == "[DONE]":
                    logger.info("Streaming completed")
                    break

                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield line
        except Exception as exc:
            logger.error("Error processing streaming response: %s", str(exc))
            raise

    def run_workflow(self, workflow_data: Dict[str, Any], stream: bool = True) -> Any:
        response = self.send_request(workflow_data, stream=stream)
        return self.process_response(response, stream=stream)

    def health_check(self) -> Dict[str, Any]:
        missing = []
        if not self.webhook_url:
            missing.append("webhook_url")
        if missing:
            return self._missing_config_result(self.webhook_url, missing)
        return self._health_check_request(self.webhook_url, headers=self.headers, timeout=5)

    def get_connection_info(self) -> Dict[str, Any]:
        return {
            "platform": self.name,
            "endpoint": self.webhook_url,
            "timeout": self.timeout,
            "has_api_key": bool(self.api_key),
        }


class DifyAgent(AgentWorkflow):
    """
    Dify SSE Agent
    Handles Server-Sent Events (SSE) API requests
    """

    def __init__(self, api_url: str, api_key: str, timeout: int = 60):
        super().__init__("Dify")
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout

        self.headers = {
            "Authorization": f"Bearer {api_key}" if api_key else "",
            "Content-Type": "application/json",
        }

    def send_request(
        self,
        query: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        response_mode: str = "streaming",
    ) -> requests.Response:
        payload = {
            "query": query,
            "user": user_id,
            "response_mode": response_mode,
            "inputs": inputs or {},
        }

        if conversation_id:
            payload["conversation_id"] = conversation_id

        try:
            logger.info("Sending request to Dify API: %s", self.api_url)
            response = self.session.post(
                self.api_url,
                json=payload,
                headers={k: v for k, v in self.headers.items() if v},
                stream=(response_mode == "streaming"),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            logger.error("Error sending request to Dify: %s", str(exc))
            raise

    def process_response(self, response: requests.Response, streaming: bool = True) -> Any:
        if streaming:
            return self._process_sse_stream(response)
        return response.json()

    def _process_sse_stream(self, response: requests.Response) -> Generator[Dict[str, Any], None, None]:
        if sseclient is None:
            yield {"type": "error", "message": "sseclient package is not installed"}
            return

        try:
            client = sseclient.SSEClient(response)

            for event in client.events():
                if not event.data:
                    continue

                if event.data.strip() in ["", "ping", ":"]:
                    continue

                try:
                    data = json.loads(event.data)
                    event_type = data.get("event", "")

                    if event_type == "message":
                        yield {
                            "type": "message",
                            "content": data.get("answer", ""),
                            "conversation_id": data.get("conversation_id"),
                        }
                    elif event_type == "message_end":
                        yield {
                            "type": "message_end",
                            "metadata": data.get("metadata", {}),
                            "conversation_id": data.get("conversation_id"),
                        }
                        logger.info("SSE streaming completed")
                        break
                    elif event_type == "error":
                        logger.error("Dify error: %s", data.get("message"))
                        yield {"type": "error", "message": data.get("message", "Unknown error")}
                        break
                    elif event_type in ("agent_message", "agent_thought"):
                        yield {
                            "type": event_type,
                            "content": data.get("thought", "") or data.get("message", ""),
                            "conversation_id": data.get("conversation_id"),
                        }
                    else:
                        if "type" not in data:
                            data["type"] = event_type or "unknown"
                        yield data

                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Failed to parse SSE data: %s... Error: %s",
                        event.data[:100],
                        str(exc),
                    )
                    yield {"type": "text", "content": event.data}
                    continue
                except Exception as exc:
                    logger.error("Error processing SSE event: %s", str(exc))
                    continue

        except Exception as exc:
            logger.error("Error processing SSE stream: %s", str(exc))
            yield {"type": "error", "message": str(exc)}

    def chat(
        self,
        query: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        streaming: bool = True,
    ) -> Any:
        response_mode = "streaming" if streaming else "blocking"
        response = self.send_request(
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
            inputs=inputs,
            response_mode=response_mode,
        )
        return self.process_response(response, streaming=streaming)

    def health_check(self) -> Dict[str, Any]:
        missing = []
        if not self.api_url:
            missing.append("api_url")
        if not self.api_key:
            missing.append("api_key")
        if missing:
            return self._missing_config_result(self.api_url, missing)
        return self._health_check_request(
            self.api_url,
            headers={k: v for k, v in self.headers.items() if v},
            timeout=5,
        )

    def get_connection_info(self) -> Dict[str, Any]:
        return {
            "platform": self.name,
            "endpoint": self.api_url,
            "timeout": self.timeout,
            "has_api_key": bool(self.api_key),
        }


class LangFlowAgent(AgentWorkflow):
    """LangFlow API agent (generic REST integration)."""

    def __init__(self, api_url: str, api_key: Optional[str] = None, timeout: int = 60):
        super().__init__("LangFlow")
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    def send_request(self, data: Dict[str, Any]) -> requests.Response:
        try:
            logger.info("Sending request to LangFlow API: %s", self.api_url)
            response = self.session.post(
                self.api_url,
                json=data,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            logger.error("Error sending request to LangFlow: %s", str(exc))
            raise

    def process_response(self, response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.text

    def chat(self, query: str, session_id: Optional[str] = None, extra_inputs: Optional[Dict[str, Any]] = None) -> Any:
        payload = {"input_value": query}
        if session_id:
            payload["session_id"] = session_id
        if extra_inputs:
            payload.update(extra_inputs)
        response = self.send_request(payload)
        return self.process_response(response)

    def health_check(self) -> Dict[str, Any]:
        missing = []
        if not self.api_url:
            missing.append("api_url")
        if missing:
            return self._missing_config_result(self.api_url, missing)
        return self._health_check_request(self.api_url, headers=self.headers, timeout=5)

    def get_connection_info(self) -> Dict[str, Any]:
        return {
            "platform": self.name,
            "endpoint": self.api_url,
            "timeout": self.timeout,
            "has_api_key": bool(self.api_key),
        }


class FlowiseAgent(AgentWorkflow):
    """Flowise API agent (generic REST integration)."""

    def __init__(self, api_url: str, api_key: Optional[str] = None, timeout: int = 60):
        super().__init__("Flowise")
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def send_request(self, data: Dict[str, Any]) -> requests.Response:
        try:
            logger.info("Sending request to Flowise API: %s", self.api_url)
            response = self.session.post(
                self.api_url,
                json=data,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            logger.error("Error sending request to Flowise: %s", str(exc))
            raise

    def process_response(self, response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.text

    def chat(self, query: str, session_id: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Any:
        payload = {"question": query}
        if session_id:
            payload["sessionId"] = session_id
        if overrides:
            payload.update(overrides)
        response = self.send_request(payload)
        return self.process_response(response)

    def health_check(self) -> Dict[str, Any]:
        missing = []
        if not self.api_url:
            missing.append("api_url")
        if missing:
            return self._missing_config_result(self.api_url, missing)
        return self._health_check_request(self.api_url, headers=self.headers, timeout=5)

    def get_connection_info(self) -> Dict[str, Any]:
        return {
            "platform": self.name,
            "endpoint": self.api_url,
            "timeout": self.timeout,
            "has_api_key": bool(self.api_key),
        }
