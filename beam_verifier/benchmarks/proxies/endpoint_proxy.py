"""Host-side proxy for model requests from agents running in sandboxes.

The proxy uses the host's network access and an optional API key to reach the
model endpoint. It listens on an address accessible only from the host, such as
`127.0.0.1`, and has no authentication of its own. Remote sandboxes reach it
through an authenticated tunnel.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import errno
import functools
import http.client
import http.cookiejar
import ipaddress
import json
import pathlib
import threading
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from typing import IO

import httpx
from aiohttp import web

DEFAULT_PORT = 8900
SANDBOX_MODEL_URL = f"http://127.0.0.1:{DEFAULT_PORT}/v1"

# Headers that describe one hop and must not be copied to the next.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)

# An agent resends its whole transcript each step; uncapped capture outgrows the run.
_CAPTURE_BODY_LIMIT = 4096

# Upstream cookies are host-side credentials; they must not reach sandbox code.
_CREDENTIAL_RESPONSE_HEADERS = frozenset({"set-cookie"})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Rejects redirects so the model API key stays at the configured endpoint."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl  # Unused.
        return None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Rejects ambiguous JSON before adding host credentials."""
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON field")
    return result


class EndpointProxy:
    """A host-side proxy that runs inside a `with` block.

    Checks each request's model before forwarding its original bytes. Streams
    responses and preserves status codes. Uses the host's trusted certificates,
    rejects redirects, never resends requests, and neither saves nor forwards
    response cookies. Exit cancels pending requests and closes connections.
    """

    def __init__(
        self,
        model_base_url: str,
        *,
        model_name: str,
        api_key: str | None = None,
        address: str = "127.0.0.1",
        port: int | None = None,
        max_connections: int | None = None,
        max_request_bytes: int = 64 * 1024 * 1024,
        capture_path: pathlib.Path | None = None,
    ) -> None:
        """Configures the host-side proxy without starting it.

        Args:
            model_base_url: Model endpoint URL, with or without a trailing `/v1`.
            model_name: The evaluated model, required in each chat request.
            api_key: Replaces the request's Authorization header when set.
                Otherwise, the header is forwarded unchanged.
            address: IP address accessible only from the host, such as `127.0.0.1`.
            port: `None` tries `DEFAULT_PORT`, then any free port; `0` uses any
                free port. Read `port` after entry to get the chosen value.
            max_connections: Maximum open connections to the model endpoint;
                `None` uses HTTPX's limit. Requests fail if all connections are busy.
            max_request_bytes: Maximum chat body size; larger requests return HTTP 413.
            capture_path: Optional JSONL file for request logs.

        Raises:
            ValueError: If the address allows remote access or a request or
                connection limit is not positive.
        """
        if not ipaddress.ip_address(address).is_loopback:
            raise ValueError("The model relay must bind to a loopback address")
        if max_connections is not None and max_connections < 1:
            raise ValueError("max_connections must be positive")
        if max_request_bytes < 1:
            raise ValueError("max_request_bytes must be positive")
        self._max_request_bytes = max_request_bytes
        self.model_base_url = model_base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.address = address
        self.port = port if port is not None else DEFAULT_PORT
        self._allow_port_fallback = port is None
        self._max_connections = max_connections
        self.http_endpoint_client = urllib.request.build_opener(
            _NoRedirect(), urllib.request.HTTPSHandler(context=httpx.create_ssl_context())
        )
        self._capture_path = capture_path
        self._capture_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._server_error: Exception | None = None

    async def _relay(
        self, request: web.Request, *, relay_client: httpx.AsyncClient, upload_lock: asyncio.Lock
    ) -> web.StreamResponse:
        methods = {"/v1/models": "GET", "/v1/chat/completions": "POST"}
        if request.path not in methods:
            raise web.HTTPNotFound()
        if request.method != methods[request.path]:
            raise web.HTTPMethodNotAllowed(request.method, (methods[request.path],))
        if request.query_string or request.headers.get("Content-Encoding", "identity") != "identity":
            raise web.HTTPBadRequest(text="Query parameters and encoded bodies are not supported")
        try:
            upstream_url = self.upstream_url(request.raw_path)
        except ValueError:
            return web.json_response({"error": "invalid proxy request target"}, status=400)
        started = time.monotonic()
        status = 502
        body = bytearray()
        body_size = 0
        owns_upload = False

        async def upload() -> AsyncIterator[bytes]:
            nonlocal body, owns_upload
            try:
                for offset in range(0, len(body), 65536):
                    yield bytes(body[offset : offset + 65536])
            finally:
                body = body[:_CAPTURE_BODY_LIMIT]
                if owns_upload:
                    owns_upload = False
                    upload_lock.release()

        response: web.StreamResponse | None = None
        model_response: httpx.Response | None = None
        response_finished = False
        try:
            if request.method == "POST":
                if request.content_length is not None and request.content_length > self._max_request_bytes:
                    raise web.HTTPRequestEntityTooLarge(self._max_request_bytes, request.content_length)
                # Bound aggregate upload memory, then release it before waiting for
                # the model response. Other requests retain only HTTP read buffers.
                await upload_lock.acquire()
                owns_upload = True
                async for chunk in request.content.iter_chunked(65536):
                    body_size += len(chunk)
                    if body_size > self._max_request_bytes:
                        raise web.HTTPRequestEntityTooLarge(self._max_request_bytes, body_size)
                    body.extend(chunk)
                try:
                    payload = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_object)
                except (ValueError, RecursionError):
                    status = 400
                    return web.json_response({"error": "Expected an unambiguous JSON object"}, status=status)
                if not isinstance(payload, dict) or payload.get("model") != self.model_name:
                    status = 403
                    return web.json_response({"error": "Request model must match the evaluated model"}, status=status)
                del payload
            elif request.can_read_body:
                status = 400
                return web.json_response({"error": "Model discovery does not accept a body"}, status=status)
            headers = {key: value for key, value in request.headers.items() if key.lower() not in _HOP_BY_HOP}
            if request.content_length is not None:
                headers["Content-Length"] = str(request.content_length)
            # Drop the sandbox's credential rather than keep a differently cased copy.
            if self.api_key:
                headers = {key: value for key, value in headers.items() if key.lower() != "authorization"}
                headers["Authorization"] = f"Bearer {self.api_key}"
            model_response = await relay_client.send(
                relay_client.build_request(
                    request.method,
                    upstream_url,
                    content=upload() if owns_upload else b"",
                    headers=headers,
                ),
                stream=True,
            )
            if 300 <= model_response.status_code < 400:
                return web.json_response(
                    {"error": "endpoint proxy: redirects are disabled; use the final model base URL"}, status=502
                )
            response = web.StreamResponse(
                status=model_response.status_code,
                headers=[
                    (key, value)
                    for key, value in model_response.headers.multi_items()
                    if key.lower() not in _HOP_BY_HOP and key.lower() not in _CREDENTIAL_RESPONSE_HEADERS
                ],
            )
            if (length := model_response.headers.get("Content-Length")) is not None:
                response.content_length = int(length)
            await response.prepare(request)
            async for chunk in model_response.aiter_raw():
                await response.write(chunk)
            await response.write_eof()
            response_finished = True
            status = model_response.status_code
            return response
        except web.HTTPRequestEntityTooLarge:
            status = 413
            raise
        except asyncio.CancelledError:
            status = 499  # Capture-only: the caller disconnected; no response is sent.
            raise
        except httpx.PoolTimeout:
            status = 503
            response = web.json_response(
                {"error": "endpoint proxy: upstream request capacity exhausted"}, status=status
            )
            response.force_close()
            return response
        except Exception as error:  # noqa: BLE001 - isolation point for one request
            if response is not None and response.prepared:
                # Leaves a truncated stream, not a second response or a clean final chunk.
                if request.transport is not None:
                    request.transport.close()
                return response
            # Exception text can embed host-side request details, credentials included;
            # only the type crosses into the sandbox.
            return web.json_response(
                {"error": f"endpoint proxy: upstream request failed: {type(error).__name__}"}, status=502
            )
        finally:
            if owns_upload:
                body = body[:_CAPTURE_BODY_LIMIT]
                owns_upload = False
                upload_lock.release()
            # A second cancellation must not interrupt the upstream socket close.
            cancelled: asyncio.CancelledError | None = None
            if model_response is not None:
                closing = asyncio.create_task(model_response.aclose())
                while not closing.done():
                    try:
                        await asyncio.shield(closing)
                    except asyncio.CancelledError as error:
                        if not response_finished:
                            status = 499
                        cancelled = error
                closing.result()
            self.record(
                method=request.method,
                path=request.raw_path,
                status=status,
                duration_s=time.monotonic() - started,
                body=bytes(body[:_CAPTURE_BODY_LIMIT]),
                body_size=body_size,
            )
            if cancelled is not None:
                raise cancelled

    def _run_server(self, ready: concurrent.futures.Future[tuple[asyncio.AbstractEventLoop, asyncio.Event]]) -> None:
        """Runs the host-side proxy until stopped, then closes its connections."""

        async def serve() -> None:
            stop = asyncio.Event()
            limits = httpx.Limits()
            if self._max_connections is not None:
                limits = httpx.Limits(
                    max_connections=self._max_connections, max_keepalive_connections=self._max_connections
                )
            async with contextlib.AsyncExitStack() as cleanup:
                relay_client = await cleanup.enter_async_context(
                    httpx.AsyncClient(
                        # Mirror owns request deadlines; a relay timeout would change its retry behavior.
                        timeout=httpx.Timeout(None, pool=0),
                        verify=httpx.create_ssl_context(),
                        follow_redirects=False,
                        limits=limits,
                        headers={"Accept-Encoding": "identity"},
                        # The relay must not retain cookies across independent trials.
                        cookies=http.cookiejar.CookieJar(policy=http.cookiejar.DefaultCookiePolicy(allowed_domains=[])),
                    )
                )
                app = web.Application(client_max_size=self._max_request_bytes)
                app.router.add_route(
                    "*",
                    "/{path:.*}",
                    functools.partial(self._relay, relay_client=relay_client, upload_lock=asyncio.Lock()),
                )
                runner = web.AppRunner(
                    app,
                    handler_cancellation=True,
                    auto_decompress=False,
                    access_log=None,
                    shutdown_timeout=0.1,
                )
                cleanup.push_async_callback(runner.cleanup)
                await runner.setup()
                try:
                    await web.TCPSite(runner, self.address, self.port).start()
                except OSError as error:
                    if not self._allow_port_fallback or error.errno != errno.EADDRINUSE:
                        raise
                    await web.TCPSite(runner, self.address, 0).start()
                self.port = int(runner.addresses[0][1])
                ready.set_result((asyncio.get_running_loop(), stop))
                await stop.wait()

        try:
            asyncio.run(serve())
        except Exception as error:  # noqa: BLE001 - propagate background lifecycle errors to the context owner
            self._server_error = error
            if not ready.done():
                ready.set_exception(error)

    @property
    def container_url(self) -> str:
        """The host-side proxy's URL, ending in `/v1`, available after entry.

        Agents in sandboxes reach this proxy through the connector's model port.
        """
        address = f"[{self.address}]" if ":" in self.address else self.address
        return f"http://{address}:{self.port}/v1"

    def upstream_url(self, path: str) -> str:
        """Builds the model endpoint URL for a request path.

        Args:
            path: Path starting with a single `/`, such as `/v1/chat/completions`.

        Returns:
            The full model endpoint URL, without duplicating `/v1`.

        Raises:
            ValueError: If the path does not start with `/` or starts with `//`.
        """
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("invalid proxy request target: expected an origin-relative path")
        return self.model_base_url.removesuffix("/v1") + path

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
        """Logs request details, a body preview, and the body bytes received."""
        if self._capture_path is None:
            return
        entry = {
            "method": method,
            "path": path,
            "status": status,
            "duration_ms": round(duration_s * 1000),
            "request_body": (body or b"")[:_CAPTURE_BODY_LIMIT].decode("utf-8", "replace"),
            "request_body_bytes": len(body or b"") if body_size is None else body_size,
        }
        with self._capture_lock, open(self._capture_path, "a") as handle:
            handle.write(json.dumps(entry) + "\n")

    def __enter__(self) -> EndpointProxy:
        """Starts the host-side proxy in a background thread.

        Raises:
            OSError: If the proxy cannot listen on the requested address and port.
        """
        if self._capture_path is not None:
            self._capture_path.parent.mkdir(parents=True, exist_ok=True)
        ready: concurrent.futures.Future[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = concurrent.futures.Future()
        self._server_error = None
        self._thread = threading.Thread(target=self._run_server, args=(ready,), name="endpoint-proxy", daemon=True)
        self._thread.start()
        try:
            self._loop, self._stop = ready.result()
        except BaseException:
            # Startup may finish binding just after the caller stops waiting.
            concurrent.futures.wait([ready])
            if ready.exception() is None:
                loop, stop = ready.result()
                loop.call_soon_threadsafe(stop.set)
            self._thread.join()
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stops the host-side proxy and closes pending model connections."""
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("Endpoint proxy did not finish shutdown within 10 seconds")
        self._loop = None
        self._stop = None
        self._thread = None
        if self._server_error is not None:
            raise self._server_error

    def check_upstream(self) -> str | None:
        """Checks whether the host can reach the model endpoint's `/v1/models`.

        Accepts 404 and 405 because model listing is optional. Does not test
        chat completions or the sandbox tunnel.

        Returns:
            A short error description, or `None` if the check succeeds.
        """
        request = urllib.request.Request(self.upstream_url("/v1/models"), method="GET")
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with self.http_endpoint_client.open(request, timeout=30):
                return None
        except urllib.error.HTTPError as error:
            with error:
                return None if error.code in (404, 405) else f"HTTP {error.code}"
        except Exception as error:  # noqa: BLE001 - any failure to reach it is the same finding
            return type(error).__name__
