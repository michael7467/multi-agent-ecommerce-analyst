from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.buy_decision_service import BuyDecisionService
from app.services.price_check_client import PriceCheckClient
from app.config.settings import settings
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.buy_decision")


class BuyDecisionAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="BuyDecisionAgent")
        self.service = BuyDecisionService()

        # Optional, opt-in by configuration -- if neither transport is
        # actually configured, this stays None and run() below behaves
        # exactly as it did before this change, no attempt, no added
        # latency. Only starts actually checking real prices once a real
        # external MCP server is configured, which hasn't happened yet
        # -- these are still placeholder settings, not a working
        # connection to anything.
        transport = getattr(settings, "price_check_transport", "http")
        http_configured = transport == "http" and getattr(settings, "price_check_mcp_server_url", None)
        stdio_configured = transport == "stdio" and getattr(settings, "price_check_command", None)

        self.price_check_client: PriceCheckClient | None = None
        if http_configured or stdio_configured:
            self.price_check_client = PriceCheckClient(
                transport=transport,
                server_url=settings.price_check_mcp_server_url,
                command=getattr(settings, "price_check_command", None),
                args=getattr(settings, "price_check_args", None),
                env=getattr(settings, "price_check_env", None),
                tool_name=getattr(settings, "price_check_tool_name", "check_price"),
                query_arg_name=getattr(settings, "price_check_query_arg_name", "query"),
                price_field_name=getattr(settings, "price_check_price_field_name", "current_price"),
            )

    @traced_agent("BuyDecisionAgent.run")
    def run(self, analysis_result: dict) -> dict:
        # BuyDecisionService itself is completely unmodified -- verified
        # directly by reading it that raw price only ever affects the
        # displayed summary text ("{title} at ${price} is
        # recommended..."), never _decide_label()'s actual
        # recommended/not-recommended/conditionally-recommended logic.
        # So the only thing worth doing here is handing the service a
        # more accurate price before it runs, not touching its
        # carefully-tuned decision logic at all.
        title = analysis_result.get("title", "")

        if self.price_check_client and title:
            current_price = self.price_check_client.check_current_price(title)

            if current_price is not None:
                logger.info(
                    "Real-time price check succeeded, using current price instead of static dataset value",
                    extra={"title": title, "static_price": analysis_result.get("price"), "current_price": current_price},
                )
                # Shallow copy, not an in-place mutation -- analysis_result
                # is the shared LangGraph state dict; overwriting it directly
                # would leak this agent's own concern into every other
                # agent reading the same object.
                analysis_result = {**analysis_result, "price": current_price}
            # current_price is None (unreachable server, schema mismatch,
            # anything) -- analysis_result is left completely untouched,
            # falling through to the original static price, exactly the
            # graceful-degradation contract price_check_client.py was
            # built and tested for.

        decision_result = self.service.make_decision(analysis_result)
        return {"buy_decision": decision_result}