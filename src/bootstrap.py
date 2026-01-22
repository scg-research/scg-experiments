import sys
from src.rag_engine import GraphRAG
from src.upload_to_neo4j import Neo4jUploader

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")


def main():
    print("=== Semantic Graph RAG Bootstrap ===\n")

    print("[1/2] Loading GraphRAG...")
    rag = GraphRAG(data_dir="data/glide", code_dir="code/glide-4.5.0")
    print(
        f"      {rag.graph.number_of_nodes()} nodes, {rag.graph.number_of_edges()} edges\n"
    )

    print("[2/2] Uploading to Neo4j...")
    embedding_map = {}
    if rag.embeddings is not None:
        vecs = rag.embeddings.cpu().numpy()
        for idx, node_id in enumerate(rag.node_ids_list):
            embedding_map[node_id] = vecs[idx].tolist()

    nodes_payload = []
    for node_id, meta in rag.node_metadata.items():
        src = rag.get_node_source(node_id, context_padding=2)
        payload = {
            "id": node_id,
            "kind": meta.get("kind", "Unknown"),
            "displayName": meta.get("display_name", "") or node_id,
            "source": src if src else "",
            "embedding": embedding_map.get(node_id, []),
        }
        if loc := meta.get("location"):
            payload["uri"] = str(loc.uri)
            payload["startLine"] = loc.startLine
            payload["endLine"] = loc.endLine
        else:
            payload["uri"] = ""
            payload["startLine"] = -1
            payload["endLine"] = -1
        if props := meta.get("properties"):
            for k, v in props.items():
                payload[f"prop_{k}"] = v
        nodes_payload.append(payload)

    edges_by_type = {}
    for u, v, data in rag.graph.edges(data=True):
        etype = data.get("type", "RELATED_TO")
        edge_payload = {"source": u, "target": v}
        if loc := data.get("location"):
            if hasattr(loc, "uri"):
                edge_payload["locUri"] = loc.uri
                edge_payload["locLine"] = loc.startLine
            elif isinstance(loc, dict):
                edge_payload["locUri"] = loc.get("uri", "")
                edge_payload["locLine"] = loc.get("start_line", -1)
        edges_by_type.setdefault(etype, []).append(edge_payload)

    try:
        uploader = Neo4jUploader(NEO4J_URI, NEO4J_AUTH)
        uploader.clear_database()
        uploader.create_indexes()
        uploader.upload_nodes(nodes_payload)
        uploader.upload_edges(edges_by_type)
        uploader.close()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Neo4j is running: docker-compose up -d")
        sys.exit(1)

    print("\n=== Done! Start MCP server with: ===")


if __name__ == "__main__":
    main()
