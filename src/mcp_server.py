import asyncio
import sys
import logging
import contextlib
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.rag_engine import GraphRAG

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

rag: GraphRAG | None = None


@contextlib.contextmanager
def redirect_stdout_to_stderr():
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


def get_rag() -> GraphRAG:
    global rag
    if rag is None:
        log.info("Initializing GraphRAG engine...")
        with redirect_stdout_to_stderr():
            rag = GraphRAG(data_dir="data/glide", code_dir="code/glide-4.5.0")
        log.info("GraphRAG engine initialized.")
    return rag


server = Server("semantic-graph-rag")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_code",
            description="Search for code entities (classes, methods, fields) in the codebase using semantic search. Returns the most relevant nodes matching your query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search for code entities (e.g., 'image loading', 'cache management', 'bitmap decoder')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_node_context",
            description="Get the context subgraph around specific code nodes, including related entities and their relationships. Useful for understanding how a piece of code connects to the rest of the codebase.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of node IDs to get context for (obtained from search_code)",
                    },
                    "hops": {
                        "type": "integer",
                        "description": "Number of relationship hops to traverse (default: 1)",
                        "default": 1,
                    },
                    "include_source": {
                        "type": "boolean",
                        "description": "Whether to include source code snippets (default: true)",
                        "default": True,
                    },
                },
                "required": ["node_ids"],
            },
        ),
        Tool(
            name="get_node_source",
            description="Get the source code for a specific node. Returns the actual code implementation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The ID of the node to get source code for",
                    },
                    "context_padding": {
                        "type": "integer",
                        "description": "Number of lines of context to include before and after (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["node_id"],
            },
        ),
        Tool(
            name="get_graph_stats",
            description="Get statistics about the loaded code graph, including node and edge counts.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    engine = get_rag()

    if name == "search_code":
        query = arguments["query"]
        limit = arguments.get("limit", 5)

        results = engine.find_nodes(query, limit=limit)

        if not results:
            return [
                TextContent(type="text", text=f"No results found for query: '{query}'")
            ]

        output = [f"Found {len(results)} results for '{query}':\n"]
        for i, node in enumerate(results, 1):
            score = node.get("score", "N/A")
            kind = node.get("kind", "Unknown")
            name = node.get("display_name") or node["id"]
            output.append(f"{i}. [{kind}] {name}")
            output.append(f"   ID: {node['id']}")
            if isinstance(score, float):
                output.append(f"   Score: {score:.4f}")
            output.append("")

        return [TextContent(type="text", text="\n".join(output))]

    elif name == "get_node_context":
        node_ids = arguments["node_ids"]
        hops = arguments.get("hops", 1)
        include_source = arguments.get("include_source", True)

        subgraph = engine.get_context_subgraph(node_ids, hops=hops)

        if include_source:
            context = engine.format_context_for_llm(subgraph, code_context_padding=3)
        else:
            output = [
                f"Context subgraph ({len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges):\n"
            ]
            output.append("Nodes:")
            for node in subgraph["nodes"]:
                kind = node.get("kind", "Unknown")
                name = node.get("display_name") or node["id"]
                output.append(f"  - [{kind}] {name} (ID: {node['id']})")

            output.append("\nRelationships:")
            for edge in subgraph["edges"]:
                s = engine.node_metadata.get(edge["source"], {}).get(
                    "display_name", edge["source"]
                )
                t = engine.node_metadata.get(edge["target"], {}).get(
                    "display_name", edge["target"]
                )
                output.append(f"  - {s} --[{edge['type']}]--> {t}")

            context = "\n".join(output)

        return [TextContent(type="text", text=context)]

    elif name == "get_node_source":
        node_id = arguments["node_id"]
        padding = arguments.get("context_padding", 5)

        source = engine.get_node_source(node_id, context_padding=padding)

        if source is None:
            meta = engine.node_metadata.get(node_id)
            if meta is None:
                return [
                    TextContent(
                        type="text", text=f"Node '{node_id}' not found in the graph."
                    )
                ]
            return [
                TextContent(
                    type="text", text=f"Source code not available for node '{node_id}'."
                )
            ]

        meta = engine.node_metadata.get(node_id, {})
        name = meta.get("display_name") or node_id
        kind = meta.get("kind", "Unknown")

        output = [
            f"Source code for [{kind}] {name}:",
            f"Node ID: {node_id}",
            "-" * 60,
            source,
            "-" * 60,
        ]

        return [TextContent(type="text", text="\n".join(output))]

    elif name == "get_graph_stats":
        graph = engine.graph
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()

        # Count nodes by kind
        kind_counts: dict[str, int] = {}
        for _, meta in engine.node_metadata.items():
            kind = meta.get("kind", "Unknown")
            kind_counts[kind] = kind_counts.get(kind, 0) + 1

        output = [
            "Graph Statistics:",
            f"  Total Nodes: {node_count:,}",
            f"  Total Edges: {edge_count:,}",
            f"  Embeddings: {'Yes' if engine.embeddings is not None else 'No'}",
            "",
            "Nodes by Kind:",
        ]

        for kind, count in sorted(kind_counts.items(), key=lambda x: -x[1]):
            output.append(f"  {kind}: {count:,}")

        return [TextContent(type="text", text="\n".join(output))]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    log.info("Starting Semantic Graph RAG MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )
    log.info("Server stopped.")


if __name__ == "__main__":
    asyncio.run(main())
