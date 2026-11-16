import ssl

import httpx
import httpx2
import pytest

from astrbot.core.utils import network_utils


def test_create_proxy_client_reuses_shared_ssl_context(
    monkeypatch: pytest.MonkeyPatch,
):
    captured_calls: list[dict] = []
    headers = {"X-Test-Header": "value"}

    class _FakeAsyncClient:
        def __init__(self, **kwargs):
            captured_calls.append(kwargs)

    monkeypatch.setattr(network_utils.httpx2, "AsyncClient", _FakeAsyncClient)

    network_utils.create_proxy_client("OpenAI")
    network_utils.create_proxy_client("OpenAI", proxy="http://127.0.0.1:7890")
    network_utils.create_proxy_client("OpenAI", headers=headers)
    network_utils.create_proxy_client("OpenAI", proxy="")

    assert len(captured_calls) == 4
    assert all(call["http2"] is True for call in captured_calls)
    assert "proxy" not in captured_calls[0]
    assert captured_calls[1]["proxy"] == "http://127.0.0.1:7890"
    assert captured_calls[2]["headers"] is headers
    assert "proxy" not in captured_calls[3]
    assert isinstance(captured_calls[0]["verify"], ssl.SSLContext)
    assert captured_calls[0]["verify"] is captured_calls[1]["verify"]
    assert captured_calls[1]["verify"] is captured_calls[2]["verify"]
    assert captured_calls[2]["verify"] is captured_calls[3]["verify"]


def test_create_proxy_client_allows_verify_override(
    monkeypatch: pytest.MonkeyPatch,
):
    captured_calls: list[dict] = []
    custom_verify = ssl.create_default_context()

    class _FakeAsyncClient:
        def __init__(self, **kwargs):
            captured_calls.append(kwargs)

    monkeypatch.setattr(network_utils.httpx2, "AsyncClient", _FakeAsyncClient)

    network_utils.create_proxy_client("OpenAI", verify=custom_verify)

    assert len(captured_calls) == 1
    assert captured_calls[0]["verify"] is custom_verify


@pytest.mark.parametrize("httpx_module", [httpx, httpx2])
@pytest.mark.parametrize("proxy", [None, "socks5://127.0.0.1:1080"])
@pytest.mark.asyncio
async def test_proxy_client_supports_http2_and_socks(httpx_module, proxy):
    async with network_utils.create_proxy_client(
        "Test", proxy=proxy, httpx_module=httpx_module
    ) as client:
        assert isinstance(client, httpx_module.AsyncClient)
        transports = [client._transport, *client._mounts.values()]
        for transport in transports:
            if transport is not None:
                assert transport._pool._http2 is True
                assert transport._pool._http1 is True


@pytest.mark.parametrize("httpx_module", [httpx, httpx2])
@pytest.mark.parametrize("error_type", ["ConnectError", "ReadTimeout", "ProxyError"])
def test_connection_error_handles_both_http_libraries(httpx_module, error_type):
    error = getattr(httpx_module, error_type)("connection failed")
    assert network_utils.is_connection_error(error)
    wrapped = RuntimeError("SDK request failed")
    wrapped.__cause__ = error
    assert network_utils.is_connection_error(wrapped)
    assert not network_utils.is_connection_error(ValueError("invalid response"))
