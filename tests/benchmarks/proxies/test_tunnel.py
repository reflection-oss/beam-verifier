"""Run tunnel tests locally using the bundled connector, without Daytona."""

import asyncio
import contextlib
import dataclasses
import json
import pathlib
import platform
from collections.abc import AsyncIterator

import aiohttp.test_utils
import aiohttp.web
import asyncssh
import httpx
import portpicker
import pytest
import pytest_asyncio

from beam_verifier.benchmarks.proxies import tunnel

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(platform.system() != "Linux", reason="Packaged sandbox executables target Linux"),
]


@dataclasses.dataclass
class Connector:
    host_key: asyncssh.SSHKey
    model_port: int
    preview_url: str
    password: str = "synthetic-tunnel-password"


@pytest_asyncio.fixture
async def connector(tmp_path: pathlib.Path) -> AsyncIterator[Connector]:
    """Run the bundled connector locally with temporary credentials and ports."""
    key = asyncssh.generate_private_key("ssh-ed25519")
    model_port = portpicker.pick_unused_port()
    websocket_port = portpicker.pick_unused_port()
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "host_key": key.export_private_key().decode(),
                "password": "synthetic-tunnel-password",
                "model_port": model_port,
                "websocket_port": websocket_port,
            }
        )
    )
    config.chmod(0o600)
    binary = tunnel.binary_path(platform.machine())
    process = await asyncio.create_subprocess_exec(str(binary), str(config), stdout=asyncio.subprocess.PIPE)
    try:
        assert process.stdout is not None
        async with asyncio.timeout(10):
            assert await process.stdout.readline() == b"SSH WebSocket connector ready\n"
        assert not config.exists(), "Connector must consume credentials before accepting connections"
        yield Connector(key, model_port, f"ws://127.0.0.1:{websocket_port}")
    finally:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        await process.wait()


async def test_forward_model_requests_preserves_bytes_and_releases_listener(connector: Connector) -> None:
    payload = b"binary\x00payload" * 20000

    async def echo(request: aiohttp.web.Request) -> aiohttp.web.Response:
        return aiohttp.web.Response(body=await request.read(), status=201)

    app = aiohttp.web.Application()
    app.router.add_post("/v1/chat/completions", echo)
    async with aiohttp.test_utils.TestServer(app, port=portpicker.pick_unused_port()) as upstream:
        async with tunnel.forward_model_requests(
            connector.preview_url,
            {},
            host_key=connector.host_key,
            password=connector.password,
            model_port=connector.model_port,
            proxy_port=upstream.port,
        ) as connection:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://127.0.0.1:{connector.model_port}/v1/chat/completions", content=payload
                )
            assert response.status_code == 201
            assert response.content == payload
            with pytest.raises(asyncssh.ChannelOpenError):
                await connection.run("id")
            with pytest.raises(asyncssh.ChannelListenError):
                await connection.forward_remote_port("0.0.0.0", connector.model_port, "127.0.0.1", upstream.port)
        assert connection.is_closed()
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.ConnectError):
                await client.get(f"http://127.0.0.1:{connector.model_port}/v1/models")
        # The same connector can serve a fresh authenticated transport.
        async with tunnel.forward_model_requests(
            connector.preview_url,
            {},
            host_key=connector.host_key,
            password=connector.password,
            model_port=connector.model_port,
            proxy_port=upstream.port,
        ) as reconnected:
            assert not reconnected.is_closed()


async def test_forward_model_requests_rejects_wrong_password(connector: Connector) -> None:
    with pytest.raises(asyncssh.PermissionDenied):
        async with tunnel.forward_model_requests(
            connector.preview_url,
            {},
            host_key=connector.host_key,
            password="wrong-synthetic-password",
            model_port=connector.model_port,
            proxy_port=portpicker.pick_unused_port(),
        ):
            pytest.fail("Wrong password authenticated")


async def test_forward_model_requests_rejects_wrong_host_key(connector: Connector) -> None:
    with pytest.raises(asyncssh.HostKeyNotVerifiable):
        async with tunnel.forward_model_requests(
            connector.preview_url,
            {},
            host_key=asyncssh.generate_private_key("ssh-ed25519"),
            password=connector.password,
            model_port=connector.model_port,
            proxy_port=portpicker.pick_unused_port(),
        ):
            pytest.fail("Untrusted SSH host authenticated")


async def test_forward_model_requests_cancellation_closes_listener(connector: Connector) -> None:
    ready = asyncio.Event()

    async def hold_tunnel() -> None:
        async with tunnel.forward_model_requests(
            connector.preview_url,
            {},
            host_key=connector.host_key,
            password=connector.password,
            model_port=connector.model_port,
            proxy_port=portpicker.pick_unused_port(),
        ):
            ready.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold_tunnel())
    try:
        async with asyncio.timeout(10):
            await ready.wait()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.ConnectError):
            await client.get(f"http://127.0.0.1:{connector.model_port}/v1/models")
