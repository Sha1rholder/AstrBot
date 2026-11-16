import asyncio
import ipaddress
import json
import ssl
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import httpx
import httpx2
import pytest
from aiohttp import web
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.events import StreamEnded

from astrbot.core.agent import mcp_client
from astrbot.core.agent.mcp_client import _create_mcp_http_client
from astrbot.core.utils.network_utils import create_proxy_client


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["h2", "http/1.1"])
async def test_http2_negotiation_and_http1_fallback(tmp_path, monkeypatch, protocol):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_path, key_path)
    server_context.set_alpn_protocols([protocol])
    client_context = ssl.create_default_context(cafile=str(cert_path))
    completed = asyncio.get_running_loop().create_future()

    async def handle(reader, writer):
        try:
            assert (
                writer.get_extra_info("ssl_object").selected_alpn_protocol() == protocol
            )
            if protocol == "h2":
                connection = H2Connection(config=H2Configuration(client_side=False))
                connection.initiate_connection()
                writer.write(connection.data_to_send())
                await writer.drain()
                while data := await reader.read(65536):
                    for event in connection.receive_data(data):
                        if isinstance(event, StreamEnded):
                            connection.send_headers(
                                event.stream_id,
                                [(":status", "200"), ("content-length", "2")],
                            )
                            connection.send_data(
                                event.stream_id, b"ok", end_stream=True
                            )
                    writer.write(connection.data_to_send())
                    await writer.drain()
            else:
                await reader.readuntil(b"\r\n\r\n")
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                await writer.drain()
                await reader.read()
        except Exception as exc:
            completed.set_exception(exc)
        else:
            completed.set_result(None)
        finally:
            writer.close()
            await writer.wait_closed()

    async with await asyncio.start_server(
        handle, "127.0.0.1", 0, ssl=server_context
    ) as server:
        port = server.sockets[0].getsockname()[1]
        async with create_proxy_client("Test", verify=client_context) as client:
            assert isinstance(client, httpx2.AsyncClient)
            response = await client.get(f"https://127.0.0.1:{port}")
            assert response.text == "ok"
            assert response.http_version == (
                "HTTP/2" if protocol == "h2" else "HTTP/1.1"
            )
        await asyncio.wait_for(completed, timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("httpx_module", [httpx, httpx2])
async def test_mcp_client_converts_sdk_timeout(httpx_module):
    timeout = httpx_module.Timeout(5, read=300, write=10, pool=None)
    async with _create_mcp_http_client(
        headers={"Authorization": "Bearer test"}, timeout=timeout
    ) as client:
        assert isinstance(client, httpx2.AsyncClient)
        assert client.timeout.as_dict() == timeout.as_dict()
        assert client.headers["Authorization"] == "Bearer test"
        assert client._transport._pool._http2 is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport", ["sse", "streamable_http", "modern_streamable_http"]
)
async def test_mcp_transports_send_requests_with_httpx2(
    monkeypatch, unused_tcp_port, transport
):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setattr(
        mcp_client, "_quick_test_mcp_connection", AsyncMock(return_value=(True, ""))
    )
    if transport == "modern_streamable_http":
        try:
            from mcp.client.streamable_http import streamable_http_client
        except ImportError:
            pytest.skip("This MCP version only provides the legacy transport API")

        monkeypatch.setattr(mcp_client, "streamable_http_client_legacy", None)
        monkeypatch.setattr(
            mcp_client, "streamable_http_client", streamable_http_client
        )
    requests = []
    events = asyncio.Queue()

    async def handle(request):
        assert request.headers["Authorization"] == "Bearer test"
        if request.method == "GET":
            if transport != "sse":
                return web.Response(status=405)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b"event: endpoint\ndata: /mcp\n\n")
            while (message := await events.get()) is not None:
                await response.write(
                    f"event: message\ndata: {json.dumps(message)}\n\n".encode()
                )
            return response
        if request.method == "DELETE":
            return web.Response(status=200)
        payload = await request.json()
        requests.append(payload["method"])
        if "id" not in payload:
            return web.Response(status=202)
        result = (
            {
                "protocolVersion": payload["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test", "version": "1.0"},
            }
            if payload["method"] == "initialize"
            else {"tools": []}
        )
        message = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
        if transport == "sse":
            await events.put(message)
            return web.Response(status=202)
        return web.json_response(message)

    app = web.Application()
    app.router.add_route("*", "/mcp", handle)
    runner = web.AppRunner(app, shutdown_timeout=1)
    await runner.setup()
    client = mcp_client.MCPClient()
    try:
        await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
        await asyncio.wait_for(
            client.connect_to_server(
                {
                    "url": f"http://127.0.0.1:{unused_tcp_port}/mcp",
                    "transport": "sse" if transport == "sse" else "streamable_http",
                    "headers": {"Authorization": "Bearer test"},
                    "timeout": 2,
                    "session_read_timeout": 2,
                },
                "test",
            ),
            timeout=5,
        )
        result = await asyncio.wait_for(client.list_tools_and_save(), timeout=5)
        assert result.tools == []
        assert requests == ["initialize", "notifications/initialized", "tools/list"]
    finally:
        await client.cleanup()
        await events.put(None)
        await runner.cleanup()
