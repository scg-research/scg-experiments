import sys
from typing import List, Dict, Any

from tqdm import tqdm
from src.rag_engine import GraphRAG
from neo4j import GraphDatabase

BATCH_SIZE = 2000


def batch_data(data: List[Any], batch_size: int = 500):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class Neo4jUploader:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.driver.verify_connectivity()
        print(f"Connected to Neo4j at {uri}")

    def close(self):
        self.driver.close()

    def clear_database(self):
        print("Clearing database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.")

    def create_indexes(self):
        print("Creating indexes...")
        with self.driver.session() as session:
            try:
                session.run("CREATE CONSTRAINT FOR (n:CodeNode) REQUIRE n.id IS UNIQUE")
            except Exception as e:
                print(f"Note: {e}")

    def upload_nodes(self, nodes_data: List[Dict]):
        print(f"Uploading {len(nodes_data)} nodes...")
        query = """
        UNWIND $batch AS row
        MERGE (n:CodeNode {id: row.id})
        SET n += row
        """
        with self.driver.session() as session:
            with tqdm(total=len(nodes_data), desc="Nodes") as pbar:
                for batch in batch_data(nodes_data, BATCH_SIZE):
                    try:
                        session.run(query, batch=batch)
                    except Exception as e:
                        print(f"Error: {e}")
                    pbar.update(len(batch))

    def _process_chunk(self, chunk):
        # Group by type within this sorted chunk to use specific queries
        by_type = {}
        for edge in chunk:
            by_type.setdefault(edge["type"], []).append(edge)

        for edge_type, edges in by_type.items():
            safe_type = (
                "".join(x for x in edge_type if x.isalnum() or x == "_").upper()
                or "RELATED_TO"
            )
            query = f"""
            UNWIND $batch AS row
            MATCH (s:CodeNode {{id: row.source}})
            MATCH (t:CodeNode {{id: row.target}})
            MERGE (s)-[r:{safe_type}]->(t)
            SET r.locUri = row.locUri,
                r.locLine = row.locLine
            """
            try:
                with self.driver.session() as session:
                    session.run(query, batch=edges)
            except Exception as e:
                print(f"Error in chunk for type {safe_type}: {e}")
        return len(chunk)

    def upload_edges(self, edges_by_type: Dict[str, List[Dict]]):
        print("Uploading edges...")

        # 1. Flatten
        all_edges = []
        for etype, edges in edges_by_type.items():
            for e in edges:
                e["type"] = etype
                all_edges.append(e)

        # 2. Sort by source to prevent Deadlocks
        all_edges.sort(key=lambda x: x["source"])

        total_edges = len(all_edges)

        # 3. Sequential Execution using sorted chunks
        chunk_size = 5000
        with tqdm(total=total_edges, desc="Edges") as pbar:
            for i in range(0, total_edges, chunk_size):
                chunk = all_edges[i : i + chunk_size]
                try:
                    count = self._process_chunk(chunk)
                    pbar.update(count)
                except Exception as e:
                    print(f"Error processing chunk starting at {i}: {e}")

        print("Finished uploading edges.")


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_AUTH = ("neo4j", "password")

    print("--- Neo4j SCG Uploader ---")
    print("Initializing GraphRAG engine...")
    rag = GraphRAG(data_dir="data/glide", code_dir="code/glide-4.5.0")

    print("Preparing data...")
    embedding_map = {}
    if rag.embeddings is not None:
        print("Extracting embeddings...")
        vecs = rag.embeddings.cpu().numpy()
        for idx, node_id in enumerate(rag.node_ids_list):
            embedding_map[node_id] = vecs[idx].tolist()

    nodes_payload = []
    print(f"Processing {len(rag.node_metadata)} nodes...")
    for node_id, meta in rag.node_metadata.items():
        src = rag.get_node_source(node_id, context_padding=2)

        # Base payload
        payload = {
            "id": node_id,
            "kind": meta.get("kind", "Unknown"),
            "displayName": meta.get("display_name", "") or node_id,
            "source": src if src else "",
            "embedding": embedding_map.get(node_id, []),
        }

        # Add Location details
        if loc := meta.get("location"):
            payload["uri"] = str(loc.uri)
            payload["startLine"] = loc.startLine
            payload["endLine"] = loc.endLine
        else:
            payload["uri"] = ""
            payload["startLine"] = -1
            payload["endLine"] = -1

        # Add dynamic properties (flattened)
        # We prefix them to avoid collision with reserved keys
        if props := meta.get("properties"):
            for k, v in props.items():
                # Neo4j properties must be primitives.
                # Assuming 'v' is string from the proto definition.
                payload[f"prop_{k}"] = v

        nodes_payload.append(payload)

    edges_by_type = {}
    print(f"Processing {rag.graph.number_of_edges()} edges...")
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
        print("\nUpload Complete! Check http://localhost:7474")
        uploader.close()
    except Exception as e:
        print(f"\nError: {e}")
        print("Ensure Docker is running: docker-compose up -d")
        sys.exit(1)


if __name__ == "__main__":
    main()
