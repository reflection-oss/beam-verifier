"""Connect agents' model requests from sandboxes to the host-side proxy.

The host opens an SSH connection to the sandbox connector over a WebSocket.
The tunnel forwards connections to one host-side proxy port. Model API keys
and sandbox-service credentials stay on the host.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import socket
from collections.abc import AsyncIterator
from importlib import metadata

import asyncssh
import httpx
from websockets.asyncio import client, connection

_SSH_USER = "beam-verifier"


def binary_path(architecture: str) -> pathlib.Path:
    """Returns the bundled connector's path for the sandbox architecture."""
    if architecture not in {"x86_64", "aarch64"}:
        raise ValueError(f"Unsupported tunnel sandbox architecture: {architecture!r}")
    package = __package__.removesuffix(".benchmarks.proxies").replace(".", "/")
    binary = pathlib.Path(
        str(
            metadata.distribution("beam-verifier").locate_file(
                f"{package}/bin/tunnel-connector/linux-{architecture}/tunnel-connector"
            )
        )
    )
    if not binary.is_file():
        raise FileNotFoundError("Tunnel connector is missing; install a beam-verifier release wheel.")
    return binary


class _PreviewConnection(client.connect):
    """Rejects redirects so preview credentials stay at the original URL."""

    def process_redirect(self, exc: Exception) -> Exception:
        return exc


@contextlib.asynccontextmanager
async def websocket_socket(websocket: connection.Connection) -> AsyncIterator[socket.socket]:
    """Lets AsyncSSH use a WebSocket connection.

    Exit closes the connections and waits for both transfer tasks to stop.

    Args:
        websocket: An open WebSocket carrying SSH data.

    Yields:
        A connected socket, closed on exit.
    """
    bridge_socket, ssh_socket = socket.socketpair()
    with bridge_socket, ssh_socket:
        reader, writer = await asyncio.open_connection(sock=bridge_socket)

        async def receive() -> None:
            try:
                async for payload in websocket:
                    if not isinstance(payload, bytes):
                        raise ValueError("SSH transport requires binary WebSocket frames")
                    writer.write(payload)
                    await writer.drain()
            finally:
                writer.close()

        async def send() -> None:
            try:
                while payload := await reader.read(65536):
                    await websocket.send(payload)
            finally:
                await websocket.close()

        pumps = [asyncio.create_task(receive()), asyncio.create_task(send())]
        try:
            yield ssh_socket
        finally:
            for pump in pumps:
                pump.cancel()
            # Transport errors reach SSH as EOF; keep its error, not the pumps'.
            await asyncio.gather(*pumps, return_exceptions=True)
            writer.close()
            await writer.wait_closed()


@contextlib.asynccontextmanager
async def forward_model_requests(
    preview_url: str,
    preview_headers: dict[str, str],
    *,
    host_key: asyncssh.SSHKey,
    password: str,
    model_port: int,
    proxy_port: int,
) -> AsyncIterator[asyncssh.SSHClientConnection]:
    """Forwards connections from a sandbox's model port to the host-side proxy.

    Start after the sandbox connector and host-side proxy are running. Exit
    closes the tunnel and stops listening on the sandbox's model port. This
    context rejects redirects and does not reconnect or resend requests.

    Args:
        preview_url: The connector's `wss://` preview URL.
        preview_headers: Headers used to authenticate with the preview service.
        host_key: SSH key used to verify the sandbox connector's identity.
        password: Temporary SSH password configured in the sandbox connector.
        model_port: Sandbox port the agent calls at `127.0.0.1`. Must be free
            and match the connector's configuration.
        proxy_port: Host-side proxy port. The sandbox cannot change this target.

    Yields:
        The SSH connection. Use `is_closed()` or `wait_closed()` to detect a
        disconnected tunnel.

    Raises:
        asyncssh.Error: If SSH authentication or connection setup fails.
        asyncssh.ChannelListenError: If the sandbox port cannot be opened.
        websockets.exceptions.WebSocketException: If the preview connection fails.
    """
    # AsyncSSH resolves the local username, which fails when the uid has no
    # passwd entry. Only host-based auth would use it; this tunnel uses a password.
    os.environ.setdefault("LOGNAME", _SSH_USER)
    async with _PreviewConnection(
        preview_url,
        additional_headers=preview_headers,
        open_timeout=20,
        ssl=httpx.create_ssl_context() if preview_url.startswith("wss://") else None,
        max_size=65536,
        max_queue=16,
        # Encrypted frames do not compress; deflate can exceed the receiver's limit.
        compression=None,
    ) as websocket:
        async with websocket_socket(websocket) as ssh_socket:
            async with asyncssh.connect(
                host="sandbox",
                sock=ssh_socket,
                config=None,
                username=_SSH_USER,
                password=password,
                client_keys=[],
                agent_path=None,
                known_hosts=([host_key.convert_to_public()], [], []),
                connect_timeout=20,
                keepalive_interval=10,
                keepalive_count_max=3,
            ) as ssh_connection:
                async with await ssh_connection.forward_remote_port("127.0.0.1", model_port, "127.0.0.1", proxy_port):
                    yield ssh_connection
