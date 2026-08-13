from __future__ import annotations

from mcp.server import MCPServer

from app.api.dependencies import get_orchestrator
from app.logging.logger import get_logger

logger = get_logger("mcp.server")

mcp_server = MCPServer(
    name="ecommerce-analyst",
    title="Multi-Agent E-commerce Analyst",
    description="Product review analysis: sentiment, trends, forecasting, recommendations, and evidence-backed reports.",
    version="1.0.0",
)


@mcp_server.tool(
    name="analyze_product",
    description=(
        "Analyzes a product using its real customer reviews. Runs sentiment "
        "analysis, retrieval, forecasting, and other analysis steps as "
        "relevant to the question asked, and returns an evidence-backed "
        "report. Same underlying analysis as this project's /analyze REST "
        "endpoint, exposed here as an MCP tool instead."
    ),
)
def analyze_product(product_id: str, query: str, top_k: int = 3) -> dict:
   
    orchestrator = get_orchestrator()

    logger.info(
        "MCP tool call: analyze_product",
        extra={"product_id": product_id, "query": query, "top_k": top_k},
    )

    result = orchestrator.run(product_id=product_id, query=query, top_k=top_k)
    return result