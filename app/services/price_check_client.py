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
    
        self.query_arg_name = query_arg_name
    
        self.price_field_name = price_field_name

    def _connect(self):
   
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
  
        try:
            return asyncio.run(self._check_current_price_async(product_title))
        except Exception:
            logger.error(
                "Price-check MCP call failed, falling back to static price data",
                exc_info=True,
                extra={"product_title": product_title},
            )
            return None