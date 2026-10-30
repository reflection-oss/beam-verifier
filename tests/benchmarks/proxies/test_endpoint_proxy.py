"""Exercise the host-side proxy against local HTTP servers."""

import asyncio
import contextlib
import errno
import http.client
import http.server
import json
import pathlib
import socket
import struct
import threading
import urllib.request
from collections.abc import AsyncIterator

import aiohttp.test_utils
import aiohttp.web
import httpx
import portpicker
import pytest

from beam_verifier.benchmarks.proxies import endpoint_proxy


@pytest.mark.parametrize("address, host", [("127.0.0.1", "127.0.0.1"), ("::1", "[::1]")])
def test_proxy_formats_bind_address(address: str, host: str) -> None:
    proxy = endpoint_proxy.EndpointProxy("https://endpoint.example.com/v1", model_name="m", address=address, port=8900)
    assert proxy.container_url == f"http://{host}:8900/v1"


def test_proxy_maps_the_request_path_onto_the_upstream_without_doubling_v1() -> None:
    proxy = endpoint_proxy.EndpointProxy(
        "https://endpoint.example.com/v1", model_name="m", address="127.0.0.1", port=1, api_key=None
    )
    assert proxy.upstream_url("/v1/chat/completions") == "https://endpoint.example.com/v1/chat/completions"
    assert proxy.container_url == "http://127.0.0.1:1/v1"


@pytest.mark.asyncio
@pytest.mark.parametrize("chunked", [False, True])
async def test_proxy_forwards_validated_request_bytes_with_bounded_capture(
    tmp_path: pathlib.Path, chunked: bool
) -> None:
    first_chunk = b'{ "model": "m", "messages": [{"role": "user", "content": "' + b"a" * 131072
    last_chunk = b"b" * 65536 + b'"}], "stream": true }'

    async def upstream(request: aiohttp.web.Request) -> aiohttp.web.Response:
        assert await request.content.readexactly(len(first_chunk)) == first_chunk
        assert await request.content.read() == last_chunk
        return aiohttp.web.json_response({"received": len(first_chunk) + len(last_chunk)})

    async def upload() -> AsyncIterator[bytes]:
        yield first_chunk
        yield last_chunk

    app = aiohttp.web.Application()
    app.router.add_post("/v1/chat/completions", upstream)
    capture = tmp_path / "proxy.jsonl"
    headers = {} if chunked else {"Content-Length": str(len(first_chunk) + len(last_chunk))}
    async with asyncio.timeout(10), aiohttp.test_utils.TestServer(app, port=portpicker.pick_unused_port()) as server:
        with endpoint_proxy.EndpointProxy(
            str(server.make_url("/v1")), model_name="m", port=0, capture_path=capture
        ) as proxy:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{proxy.container_url}/chat/completions", content=upload(), headers=headers
                )
                assert response.status_code == 200
                assert response.json() == {"received": len(first_chunk) + len(last_chunk)}
    entry = json.loads(capture.read_text())
    assert entry["request_body"] == first_chunk[:4096].decode()
    assert entry["request_body_bytes"] == len(first_chunk) + len(last_chunk)


@pytest.mark.asyncio
async def test_proxy_rejects_excess_requests_before_forwarding_and_reuses_capacity() -> None:
    received: list[str] = []
    started = asyncio.Event()
    release_response = asyncio.Event()

    async def upstream(request: aiohttp.web.Request) -> aiohttp.web.Response:
        received.append(request.method)
        started.set()
        await release_response.wait()
        return aiohttp.web.Response(text="done")

    app = aiohttp.web.Application()
    app.router.add_route("*", "/v1/models", upstream)
    async with asyncio.timeout(10), aiohttp.test_utils.TestServer(app, port=portpicker.pick_unused_port()) as server:
        with endpoint_proxy.EndpointProxy(
            str(server.make_url("/v1")), model_name="m", port=0, max_connections=1
        ) as proxy:
            async with httpx.AsyncClient() as client:
                first = asyncio.create_task(client.get(f"{proxy.container_url}/models"))
                try:
                    await started.wait()
                    excess = await client.get(f"{proxy.container_url}/models")
                    assert excess.status_code == 503
                    assert "capacity exhausted" in excess.json()["error"]
                    assert excess.headers["connection"] == "close"
                    assert received == ["GET"]
                finally:
                    release_response.set()
                    completed = await first
                assert completed.status_code == 200
                assert (await client.get(f"{proxy.container_url}/models")).status_code == 200
    assert received == ["GET", "GET"]


def test_proxy_rejects_zero_upstream_capacity() -> None:
    with pytest.raises(ValueError, match="max_connections must be positive"):
        endpoint_proxy.EndpointProxy("http://127.0.0.1:1/v1", model_name="m", max_connections=0)


@pytest.mark.asyncio
async def test_proxy_does_not_forward_incomplete_upload() -> None:
    received: list[bytes] = []

    async def upstream(request: aiohttp.web.Request) -> aiohttp.web.Response:
        received.append(await request.read())
        return aiohttp.web.Response()

    app = aiohttp.web.Application()
    app.router.add_post("/v1/chat/completions", upstream)
    async with aiohttp.test_utils.TestServer(app, port=portpicker.pick_unused_port()) as server:
        with endpoint_proxy.EndpointProxy(str(server.make_url("/v1")), model_name="m", port=0) as proxy:
            reader, writer = await asyncio.open_connection(proxy.address, proxy.port)
            writer.write(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1000\r\n\r\nabc")
            await writer.drain()
            writer.write_eof()
            await reader.read()
            writer.close()
            await writer.wait_closed()
    assert received == [], "An incomplete request reached the model endpoint"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, expected_status",
    [
        (b'{"model":"other"}', 403),
        (b"{}", 403),
        (b"[]", 403),
        (b'{"model":["m"]}', 403),
        (b'{"model":"other", "model":"m"}', 400),
        (b'{"model":"m", "model":"other"}', 400),
        (b'{"model":', 400),
    ],
)
async def test_proxy_rejects_unapproved_model_before_contacting_upstream(body: bytes, expected_status: int) -> None:
    received: list[str | None] = []

    async def upstream(request: aiohttp.web.Request) -> aiohttp.web.Response:
        received.append(request.headers.get("Authorization"))
        return aiohttp.web.Response()

    app = aiohttp.web.Application()
    app.router.add_post("/v1/chat/completions", upstream)
    async with aiohttp.test_utils.TestServer(app, port=portpicker.pick_unused_port()) as server:
        with endpoint_proxy.EndpointProxy(
            str(server.make_url("/v1")), model_name="m", api_key="synthetic-host-key", port=0
        ) as proxy:
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{proxy.container_url}/chat/completions", content=body)
    assert response.status_code == expected_status
    assert "synthetic-host-key" not in response.text
    assert received == [], "The proxy granted host access to an unapproved model"


def test_proxy_reports_the_actual_os_assigned_port() -> None:
    with endpoint_proxy.EndpointProxy("http://127.0.0.1:1/v1", model_name="m", port=0) as proxy:
        assert proxy.port > 0
        assert proxy.container_url == f"http://127.0.0.1:{proxy.port}/v1"
        with contextlib.closing(http.client.HTTPConnection(proxy.address, proxy.port, timeout=5)) as client:
            client.request("PUT", "/v1/chat/completions")
            with client.getresponse() as response:
                assert response.status == 405


def test_proxy_default_port_collision_falls_back_at_bind_time() -> None:
    # Construct both before either binds, reproducing the old probe-then-bind race.
    first = endpoint_proxy.EndpointProxy("http://127.0.0.1:1/v1", model_name="m", address="127.0.0.2")
    second = endpoint_proxy.EndpointProxy("http://127.0.0.1:1/v1", model_name="m", address="127.0.0.2")
    with first, second:
        assert first.port != second.port
        with contextlib.closing(http.client.HTTPConnection(second.address, second.port, timeout=5)) as client:
            client.request("PUT", "/v1/chat/completions")
            with client.getresponse() as response:
                assert response.status == 405


def test_proxy_does_not_fall_back_from_an_explicit_occupied_port() -> None:
    with endpoint_proxy.EndpointProxy("http://127.0.0.1:1/v1", model_name="m", port=0) as first:
        with pytest.raises(OSError) as raised:
            with endpoint_proxy.EndpointProxy("http://127.0.0.1:1/v1", model_name="m", port=first.port):
                pytest.fail("An explicitly requested occupied port must fail")
        assert raised.value.errno == errno.EADDRINUSE


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_proxy_rejects_redirects_without_forwarding_credentials(method: str) -> None:
    received: list[tuple[str, str | None]] = []
    upstream_port = portpicker.pick_unused_port()

    class RedirectingEndpoint(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server naming convention
            received.append((self.path, self.headers.get("Authorization")))
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200 if self.path == "/redirect-target" else 302)
            self.send_header("Location", f"http://localhost:{upstream_port}/redirect-target")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_POST = do_GET

    with http.server.ThreadingHTTPServer(("127.0.0.1", upstream_port), RedirectingEndpoint) as upstream:
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            with endpoint_proxy.EndpointProxy(
                f"http://127.0.0.1:{upstream_port}/v1", model_name="m", api_key="host-secret"
            ) as proxy:
                assert proxy.check_upstream() == "HTTP 302"
                connection = http.client.HTTPConnection(proxy.address, proxy.port, timeout=10)
                try:
                    connection.request(
                        method,
                        "/v1/models" if method == "GET" else "/v1/chat/completions",
                        body=None if method == "GET" else b'{"model":"m"}',
                    )
                    with connection.getresponse() as response:
                        assert response.status == 502
                        assert "redirect" in json.loads(response.read())["error"]
                finally:
                    connection.close()
        finally:
            upstream.shutdown()
            thread.join(timeout=10)

    assert received == [
        ("/v1/models", "Bearer host-secret"),
        ("/v1/models" if method == "GET" else "/v1/chat/completions", "Bearer host-secret"),
    ]


@pytest.mark.parametrize(
    ("path", "error_message"),
    [
        ("@localhost:1234/v1/models", "Unexpected start char in url"),
        ("http://localhost:1234/v1/models", "invalid proxy request target"),
    ],
)
def test_proxy_rejects_non_origin_request_targets(path: str, error_message: str) -> None:
    with endpoint_proxy.EndpointProxy("http://127.0.0.1:1/v1", model_name="m") as proxy:
        connection = http.client.HTTPConnection(proxy.address, proxy.port, timeout=10)
        try:
            connection.request("GET", path)
            with connection.getresponse() as response:
                assert response.status == 400
                # Syntactically invalid targets are rejected by the HTTP parser
                # before the application's JSON error handler can run.
                assert error_message in response.read().decode()
        finally:
            connection.close()


@pytest.mark.parametrize("stream_started", [False, True])
def test_proxy_cancels_upstream_request_after_client_disconnect(
    tmp_path: pathlib.Path, capfd: pytest.CaptureFixture[str], stream_started: bool
):
    request_started = threading.Event()
    upstream_disconnected = threading.Event()
    request_recorded = threading.Event()
    upstream_requests: list[str] = []

    class ClosingEndpoint(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server naming convention
            upstream_requests.append(self.path)
            if stream_started:
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"3\r\nabc\r\n")
            request_started.set()
            self.connection.settimeout(5)
            assert self.connection.recv(1) == b"", "The proxy kept its upstream request open after cancellation"
            upstream_disconnected.set()
            self.close_connection = True

    class ObservedProxy(endpoint_proxy.EndpointProxy):
        def record(
            self,
            *,
            method: str,
            path: str,
            status: int,
            duration_s: float,
            body: bytes | None,
            body_size: int | None = None,
        ) -> None:
            super().record(
                method=method, path=path, status=status, duration_s=duration_s, body=body, body_size=body_size
            )
            request_recorded.set()

    capture = tmp_path / "proxy.jsonl"
    with http.server.ThreadingHTTPServer(("127.0.0.1", portpicker.pick_unused_port()), ClosingEndpoint) as upstream:
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            with ObservedProxy(
                f"http://127.0.0.1:{upstream.server_address[1]}/v1", model_name="m", capture_path=capture
            ) as proxy:
                with socket.create_connection((proxy.address, proxy.port), timeout=10) as client:
                    client.sendall(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n")
                    assert request_started.wait(timeout=10)
                    # Send RST so an attempted error response fails at its first write.
                    client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                assert upstream_disconnected.wait(timeout=5), "The pending upstream request was not cancelled"
                assert request_recorded.wait(timeout=10), "The disconnected request escaped without being recorded"
        finally:
            upstream.shutdown()
            thread.join(timeout=10)
    assert upstream_requests == ["/v1/models"], "Cancellation must not replay a request"
    assert json.loads(capture.read_text())["status"] == 499
    assert "Traceback" not in capfd.readouterr().err


def test_proxy_shutdown_cancels_pending_upstream_request() -> None:
    request_started = threading.Event()
    upstream_disconnected = threading.Event()

    class PendingEndpoint(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server naming convention
            request_started.set()
            self.connection.settimeout(5)
            assert self.connection.recv(1) == b""
            upstream_disconnected.set()

    with http.server.ThreadingHTTPServer(("127.0.0.1", portpicker.pick_unused_port()), PendingEndpoint) as upstream:
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        client = http.client.HTTPConnection("127.0.0.1", timeout=5)
        try:
            with endpoint_proxy.EndpointProxy(
                f"http://127.0.0.1:{upstream.server_address[1]}/v1", model_name="m"
            ) as proxy:
                client.port = proxy.port
                client.request("GET", "/v1/models")
                assert request_started.wait(timeout=5)
            assert upstream_disconnected.wait(timeout=5), "Proxy shutdown left the model request running"
            with pytest.raises(http.client.RemoteDisconnected):
                client.getresponse()
            with socket.socket() as probe:
                assert probe.connect_ex((proxy.address, proxy.port)) != 0
        finally:
            client.close()
            upstream.shutdown()
            thread.join(timeout=10)


def test_proxy_closes_a_broken_stream_without_sending_a_second_response() -> None:
    class BrokenStream(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server naming convention
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"3\r\nabc\r\ninvalid-chunk\r\n")
            self.close_connection = True

    with http.server.ThreadingHTTPServer(("127.0.0.1", portpicker.pick_unused_port()), BrokenStream) as upstream:
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            with endpoint_proxy.EndpointProxy(
                f"http://127.0.0.1:{upstream.server_address[1]}/v1", model_name="m"
            ) as proxy:
                with socket.create_connection((proxy.address, proxy.port), timeout=10) as client:
                    client.sendall(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n")
                    with client.makefile("rb") as response:
                        received = response.read()
        finally:
            upstream.shutdown()
            thread.join(timeout=10)
    assert received.startswith(b"HTTP/1.1 200")
    assert received.endswith(b"3\r\nabc\r\n"), "A broken stream must end without a final chunk or a second response"


def test_proxy_does_not_retain_cookies_or_duplicate_authorization() -> None:
    received: list[tuple[str | None, list[str]]] = []

    class CookieEndpoint(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server naming convention
            received.append((self.headers.get("Cookie"), self.headers.get_all("Authorization", [])))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.send_header("Set-Cookie", "upstream_session=synthetic-auth-cookie; Path=/")
            self.send_header("Set-Cookie", "second=synthetic-auth-cookie; Path=/")
            self.end_headers()

    with http.server.ThreadingHTTPServer(("127.0.0.1", portpicker.pick_unused_port()), CookieEndpoint) as upstream:
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            with endpoint_proxy.EndpointProxy(
                f"http://127.0.0.1:{upstream.server_address[1]}/v1", model_name="m", api_key="host-secret"
            ) as proxy:
                client = http.client.HTTPConnection(proxy.address, proxy.port, timeout=5)
                try:
                    client.request(
                        "GET", "/v1/models", headers={"authorization": "Bearer placeholder", "Cookie": "agent=one"}
                    )
                    with client.getresponse() as response:
                        assert response.status == 200
                        # An upstream authentication cookie is a host credential; it must not
                        # reach the sandbox.
                        assert response.headers.get_all("Set-Cookie") is None
                        response.read()
                    client.request("GET", "/v1/models")
                    with client.getresponse() as response:
                        assert response.status == 200
                        response.read()
                finally:
                    client.close()
        finally:
            upstream.shutdown()
            thread.join(timeout=10)
    assert received == [("agent=one", ["Bearer host-secret"]), (None, ["Bearer host-secret"])]


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_proxy_does_not_replay_requests_after_upstream_disconnect(method: str) -> None:
    received: list[str] = []

    class ClosingEndpoint(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server naming convention
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            received.append(self.command)
            self.close_connection = True

        do_POST = do_GET

    with http.server.ThreadingHTTPServer(("127.0.0.1", portpicker.pick_unused_port()), ClosingEndpoint) as upstream:
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            with endpoint_proxy.EndpointProxy(
                f"http://127.0.0.1:{upstream.server_address[1]}/v1", model_name="m"
            ) as proxy:
                client = http.client.HTTPConnection(proxy.address, proxy.port, timeout=5)
                try:
                    client.request(
                        method,
                        "/v1/models" if method == "GET" else "/v1/chat/completions",
                        body=None if method == "GET" else b'{"model":"m"}',
                    )
                    with client.getresponse() as response:
                        assert response.status == 502
                        response.read()
                finally:
                    client.close()
        finally:
            upstream.shutdown()
            thread.join(timeout=10)
    assert received == [method], "The proxy replayed a request which may have already executed"


@pytest.mark.parametrize(
    ("models_status", "expected_error"),
    [(200, None), (404, None), (405, None), (401, "HTTP 401"), (503, "HTTP 503")],
)
def test_proxy_checks_optional_models_route_and_relays_requests(
    tmp_path: pathlib.Path, models_status: int, expected_error: str | None
):
    seen: list[tuple[str, str | None]] = []

    class _Upstream(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - base class signature
            del format, args  # Unused.

        def do_GET(self) -> None:  # noqa: N802 - http.server naming convention
            self.send_response(models_status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802 - http.server naming convention
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            seen.append((self.path, self.headers.get("Authorization")))
            payload = b'{"id":"chatcmpl-1"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    capture = tmp_path / "proxy.jsonl"
    try:
        with endpoint_proxy.EndpointProxy(
            upstream_url, model_name="m", api_key="secret", address="127.0.0.1", capture_path=capture
        ) as proxy:
            assert proxy.check_upstream() == expected_error
            request = urllib.request.Request(
                f"{proxy.container_url}/chat/completions",
                data=b'{"model":"m"}',
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                assert json.loads(response.read())["id"] == "chatcmpl-1"
    finally:
        upstream.shutdown()
        upstream.server_close()

    # The container never holds the real key; the proxy attaches it on the way out.
    assert seen == [("/v1/chat/completions", "Bearer secret")]
    captured = json.loads(capture.read_text().strip())
    assert captured["path"] == "/v1/chat/completions"
    assert captured["status"] == 200
    assert "secret" not in capture.read_text()


def test_proxy_response_keeps_the_upstream_content_length() -> None:
    # Framing is load-bearing: the relay strips hop-by-hop headers, and a
    # response with no length and no chunked framing leaves an HTTP/1.1 client
    # waiting for a connection close that never comes.
    payload = b'{"id":"chatcmpl-1"}'

    class _Upstream(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - base class signature
            del format, args  # Unused.

        def do_POST(self) -> None:  # noqa: N802 - http.server naming convention
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    try:
        with endpoint_proxy.EndpointProxy(
            f"http://127.0.0.1:{upstream.server_address[1]}/v1", model_name="m", address="127.0.0.1"
        ) as proxy:
            connection = http.client.HTTPConnection(proxy.address, proxy.port, timeout=10)
            connection.request("POST", "/v1/chat/completions", body=b'{"model":"m"}')
            response = connection.getresponse()
            assert response.getheader("Content-Length") == str(len(payload))
            assert response.read() == payload
            connection.close()
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_proxy_reframes_an_unsized_upstream_response_as_chunked() -> None:
    # An upstream that streams (no Content-Length) is relayed with chunked
    # framing, so the client knows where the body ends without a length.
    payload = b'data: {"choices":[]}\n\ndata: [DONE]\n\n'

    class _Upstream(http.server.BaseHTTPRequestHandler):
        # HTTP/1.0: the body is delimited by connection close, never by a length.
        protocol_version = "HTTP/1.0"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - base class signature
            del format, args  # Unused.

        def do_POST(self) -> None:  # noqa: N802 - http.server naming convention
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(payload)

    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    try:
        with endpoint_proxy.EndpointProxy(
            f"http://127.0.0.1:{upstream.server_address[1]}/v1", model_name="m", address="127.0.0.1"
        ) as proxy:
            connection = http.client.HTTPConnection(proxy.address, proxy.port, timeout=10)
            connection.request("POST", "/v1/chat/completions", body=b'{"model":"m"}')
            response = connection.getresponse()
            assert response.getheader("Transfer-Encoding") == "chunked"
            assert response.read() == payload
            connection.close()
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_proxy_reports_an_unreachable_upstream_instead_of_raising() -> None:
    # Port 1 on loopback refuses connections, which is the failure a private
    # endpoint produces and which must be caught before any container starts.
    proxy = endpoint_proxy.EndpointProxy("http://127.0.0.1:1/v1", model_name="m", address="127.0.0.1", port=2)
    assert proxy.check_upstream() is not None


def test_proxy_rejects_public_bind_address() -> None:
    with pytest.raises(ValueError, match="loopback"):
        endpoint_proxy.EndpointProxy("https://endpoint.invalid/v1", model_name="m", address="0.0.0.0")


def test_proxy_rejects_unrelated_upstream_paths() -> None:
    with endpoint_proxy.EndpointProxy("http://127.0.0.1:1/v1", model_name="m", port=0) as proxy:
        response = httpx.get(f"http://127.0.0.1:{proxy.port}/admin")
    assert response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("chunked", [False, True])
async def test_proxy_rejects_oversized_upload_and_accepts_next_request(chunked: bool) -> None:
    received: list[bytes] = []

    async def upstream(request: aiohttp.web.Request) -> aiohttp.web.Response:
        received.append(await request.read())
        return aiohttp.web.Response()

    async def upload() -> AsyncIterator[bytes]:
        yield b"x" * 33

    app = aiohttp.web.Application()
    app.router.add_post("/v1/chat/completions", upstream)
    async with asyncio.timeout(10), aiohttp.test_utils.TestServer(app, port=portpicker.pick_unused_port()) as server:
        with endpoint_proxy.EndpointProxy(
            str(server.make_url("/v1")), model_name="m", port=0, max_request_bytes=32
        ) as proxy:
            async with httpx.AsyncClient() as client:
                oversized = await client.post(
                    f"{proxy.container_url}/chat/completions", content=upload() if chunked else b"x" * 33
                )
                valid = await client.post(f"{proxy.container_url}/chat/completions", json={"model": "m"})
    assert oversized.status_code == 413
    assert valid.status_code == 200
    assert received == [b'{"model":"m"}']


@pytest.mark.asyncio
async def test_proxy_releases_upload_before_model_response() -> None:
    response_ready = asyncio.Event()
    received: asyncio.Queue[bytes] = asyncio.Queue()

    async def upstream(request: aiohttp.web.Request) -> aiohttp.web.Response:
        await received.put(await request.read())
        await response_ready.wait()
        return aiohttp.web.Response()

    app = aiohttp.web.Application()
    app.router.add_post("/v1/chat/completions", upstream)
    async with asyncio.timeout(10), aiohttp.test_utils.TestServer(app, port=portpicker.pick_unused_port()) as server:
        with endpoint_proxy.EndpointProxy(str(server.make_url("/v1")), model_name="m", port=0) as proxy:
            async with httpx.AsyncClient() as client:
                first = asyncio.create_task(client.post(f"{proxy.container_url}/chat/completions", json={"model": "m"}))
                assert await received.get() == b'{"model":"m"}'
                second = asyncio.create_task(
                    client.post(f"{proxy.container_url}/chat/completions", json={"model": "m"})
                )
                try:
                    assert await received.get() == b'{"model":"m"}'
                    assert not first.done(), "The first model response must still be pending"
                finally:
                    response_ready.set()
                    results = await asyncio.gather(first, second)
                assert results[0].status_code == results[1].status_code == 200
