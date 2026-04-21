import json
import unittest
from unittest.mock import patch

from stark.providers import ClaudeCompatibleConfig, ClaudeCompatibleProvider


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class ClaudeProviderTests(unittest.TestCase):
    def test_messages_endpoint_uses_v1_messages(self):
        provider = ClaudeCompatibleProvider(
            ClaudeCompatibleConfig(
                api_key="test-key",
                base_url="https://open.xiaojingai.com",
                model="claude-sonnet-4-6",
            )
        )
        self.assertEqual(provider._messages_endpoint(), "https://open.xiaojingai.com/v1/messages")

    def test_post_json_adds_anthropic_version_header(self):
        provider = ClaudeCompatibleProvider(
            ClaudeCompatibleConfig(
                api_key="test-key",
                base_url="https://open.xiaojingai.com",
                model="claude-sonnet-4-6",
                api_version="2023-06-01",
            )
        )
        captured = {}

        def _fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _DummyResponse({"content": [{"text": "ok"}]})

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            payload = provider._post_json(
                provider._messages_endpoint(),
                {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hello"}]},
            )

        headers = {key.lower(): value for key, value in captured["request"].header_items()}
        self.assertEqual(payload["content"][0]["text"], "ok")
        self.assertEqual(captured["timeout"], provider.config.timeout_seconds)
        self.assertEqual(headers["authorization"], "Bearer test-key")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")


if __name__ == "__main__":
    unittest.main()
