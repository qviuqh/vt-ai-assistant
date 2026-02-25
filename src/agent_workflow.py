from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Generator, List
import os
import json
import logging
from pathlib import Path

import requests
import sseclient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AgentWorkflow(ABC):
    """Base class for AI Agent workflow integrations."""

    def __init__(self, name: str):
        self.name = name
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