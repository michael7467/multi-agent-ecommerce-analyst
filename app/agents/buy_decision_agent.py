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

        title = analysis_result.get("title", "")

        if self.price_check_client and title:
            current_price = self.price_check_client.check_current_price(title)

            if current_price is not None:
                logger.info(
                    "Real-time price check succeeded, using current price instead of static dataset value",
                    extra={"title": title, "static_price": analysis_result.get("price"), "current_price": current_price},
                )
          
                analysis_result = {**analysis_result, "price": current_price}


        decision_result = self.service.make_decision(analysis_result)
        return {"buy_decision": decision_result}