from __future__ import annotations

import asyncio
import json
from typing import Literal

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from app.logging.logger import get_logger

logger = get_logger("services.price_check_client")


class PriceCheckClient:
    """
    Connects to an external, configurable MCP server to check a product's
    real, current price/availability -- data the project's own static
    2023 dataset structurally cannot provide.

    Deliberately generic/configurable rather than hardcoded to one named
    third-party service: no specific external MCP server has actually
    been chosen yet, so this connects to whatever URL/command and tool
    name are passed in, rather than assuming a particular provider's API
    shape.

    Supports two transports, since real MCP servers aren't consistent
    about which one they use: "http" connects to a remote URL (the
    original design -- suits a server this project would run itself,
    reachable over the network). "stdio" spawns a local subprocess and
    talks to it over stdin/stdout -- confirmed by research to be how
    every real Keepa-backed MCP server implementation found actually
    runs, built for a desktop client to spawn locally rather than a
    remote HTTP endpoint to call.
    """

    def __init__(
        self,
        transport: Literal["http", "stdio"] = "http",
        server_url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        tool_name: str = "check_price",
        query_arg_name: str = "query",
        price_field_name: str = "current_price",
    ) -> None:
        self.transport = transport

        if transport == "http":
            if not server_url:
                raise ValueError("server_url is required when transport='http'")
        elif transport == "stdio":
            if not command:
                raise ValueError("command is required when transport='stdio'")
        else:
            raise ValueError(f"Unknown transport: {transport!r} -- must be 'http' or 'stdio'")

        self.server_url = server_url
        self.command = command
        self.args = args or []
        self.env = env

        self.tool_name = tool_name
        # Configurable, not hardcoded -- caught directly by testing:
        # different external MCP servers expect different argument
        # names for "what to look up" (query, product_name,
        # search_term, product_id...). Assuming one fixed name would
        # silently fail against any real server that doesn't happen to
        # match it.
        self.query_arg_name = query_arg_name
        # Same reasoning, applied to the response side -- different
        # servers will name the price field differently too
        # (current_price, price, cost...).
        self.price_field_name = price_field_name

    def _connect(self):
        """
        Returns the right async context manager for the configured
        transport. Deliberately the ONLY place transport choice affects
        anything -- everything downstream of this (session
        initialization, the tool call itself, response parsing) is
        identical either way, since that's the whole point of MCP's
        layered design: once you have a session, the rest of the
        protocol doesn't care how the bytes got there.
        """
        if self.transport == "http":
            return streamable_http_client(self.server_url)

        return stdio_client(
            StdioServerParameters(command=self.command, args=self.args, env=self.env)
        )

    async def _check_current_price_async(self, product_title: str) -> float | None:
        async with self._connect() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    self.tool_name, arguments={self.query_arg_name: product_title}
                )

                if result.is_error:
                    logger.warning(
                        "Price-check MCP tool returned an error",
                        extra={"product_title": product_title},
                    )
                    return None

                # Parsed into a clean number here, rather than leaving
                # raw MCP content blocks (TextContent objects, JSON-as-
                # text) leaking out to buy_decision_agent, which only
                # actually needs one float. Any failure at any step here
                # -- empty content, non-JSON text, missing field,
                # non-numeric value -- returns None rather than raising,
                # same graceful-degradation contract as the rest of this
                # class.
                try:
                    payload = json.loads(result.content[0].text)
                    return float(payload[self.price_field_name])
                except (IndexError, AttributeError, json.JSONDecodeError, KeyError, ValueError, TypeError):
                    logger.warning(
                        "Price-check MCP response could not be parsed into a price",
                        extra={"product_title": product_title, "raw_content": str(result.content)},
                    )
                    return None

    def check_current_price(self, product_title: str) -> float | None:
        """
        Synchronous entry point -- buy_decision_agent.run() is a plain
        sync method, matching every other agent's .run() throughout this
        project, so this bridges to the MCP client's async API via
        asyncio.run() rather than requiring the caller to become async.

        Returns None on ANY failure -- unreachable server, timeout,
        malformed response, anything -- rather than raising, and a clean
        float on success. This is a non-critical, best-effort
        enhancement to buy_decision, not a required dependency: the
        existing static price data is always the fallback, the same
        graceful-degradation pattern already used throughout this
        project (e.g. RAGService's cache-read handling).
        """
        try:
            return asyncio.run(self._check_current_price_async(product_title))
        except Exception:
            logger.error(
                "Price-check MCP call failed, falling back to static price data",
                exc_info=True,
                extra={"product_title": product_title},
            )
            return None