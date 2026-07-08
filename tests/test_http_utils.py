from unittest.mock import Mock, patch

from stark.providers.http_utils import post_json_request, stream_json_events_request


def test_post_json_request_uses_bounded_connect_timeout():
    response = Mock()
    response.status_code = 200
    response.text = '{"ok": true}'
    response.json.return_value = {"ok": True}

    with patch("stark.providers.http_utils.requests.post", return_value=response) as post:
        result = post_json_request(
            url="https://example.test/v1/chat/completions",
            request_body={"x": 1},
            headers={"Content-Type": "application/json"},
            timeout_seconds=600,
            error_prefix="LLM request failed",
        )

    assert result == {"ok": True}
    assert post.call_args.kwargs["timeout"] == (20.0, 600.0)


def test_post_json_request_keeps_short_timeout_for_quick_profiles():
    response = Mock()
    response.status_code = 200
    response.text = '{"ok": true}'
    response.json.return_value = {"ok": True}

    with patch("stark.providers.http_utils.requests.post", return_value=response) as post:
        post_json_request(
            url="https://example.test/v1/chat/completions",
            request_body={"x": 1},
            headers={"Content-Type": "application/json"},
            timeout_seconds=15,
            error_prefix="LLM request failed",
        )

    assert post.call_args.kwargs["timeout"] == (15.0, 15.0)


def test_stream_json_events_request_parses_sse_lines():
    response = Mock()
    response.status_code = 200
    response.text = ""
    response.iter_lines.return_value = [
        b"event: message",
        b'data: {"choices":[{"delta":{"content":"hel"}}]}',
        b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}',
        b"data: [DONE]",
    ]

    with patch("stark.providers.http_utils.requests.post", return_value=response) as post:
        result = stream_json_events_request(
            url="https://example.test/v1/chat/completions",
            request_body={"stream": True},
            headers={"Content-Type": "application/json"},
            timeout_seconds=30,
            error_prefix="LLM request failed",
        )

    assert result == [
        {"choices": [{"delta": {"content": "hel"}}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
    ]
    assert post.call_args.kwargs["stream"] is True
    assert post.call_args.kwargs["timeout"] == (20.0, 30.0)
