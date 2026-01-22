import os
import torch
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from src.models.semantic_graph_pb2 import SemanticGraphFile
from sentence_transformers import SentenceTransformer, util


class GraphRAG:
    def __init__(
        self, data_dir: Union[str, Path], code_dir: Optional[Union[str, Path]] = None
    ):
        self.data_dir = Path(data_dir)
        self.code_dir = Path(code_dir) if code_dir else None
        self.graph = nx.DiGraph()
        self.node_metadata: Dict[str, Dict[str, Any]] = {}
        self.node_ids_list: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.model = None
        self.file_map: Dict[str, Path] = {}

        print("Initializing model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        self._load_graph()
        if self.code_dir:
            self._build_code_index()
        self._link_definitions()
        if self.model:
            self._build_index()

    def _load_graph(self):
        print(f"Loading graphs from {self.data_dir}...")
        files = list(self.data_dir.rglob("*.semanticgraphdb"))
        for file_path in files:
            try:
                with open(file_path, "rb") as f:
                    sgf = SemanticGraphFile()
                    sgf.ParseFromString(f.read())
                    self._process_file(sgf)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        print(
            f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges"
        )

    def _process_file(self, sgf: SemanticGraphFile):
        file_uri = sgf.uri
        for node in sgf.nodes:
            if node.id not in self.node_metadata:
                self.graph.add_node(
                    node.id, kind=node.kind, display_name=node.displayName
                )
                self.node_metadata[node.id] = {
                    "kind": node.kind,
                    "display_name": node.displayName,
                    "location": node.location,
                    "properties": dict(node.properties),
                    "file_uri": file_uri,
                }
            for edge in node.edges:
                self.graph.add_edge(node.id, edge.to, type=edge.type)

    def _link_definitions(self):
        print("Linking definitions to files...")
        count = 0
        for node_id, meta in self.node_metadata.items():
            if loc := meta.get("location"):
                file_uri = loc.uri
                if file_uri and self.graph.has_node(file_uri) and file_uri != node_id:
                    self.graph.add_edge(file_uri, node_id, type="CONTAINS")
                    count += 1
        print(f"Created {count} CONTAINS edges.")

    def _build_code_index(self):
        print(f"Indexing source code in {self.code_dir}...")
        for root, _, files in os.walk(self.code_dir):
            for file in files:
                if file.endswith(
                    (".java", ".constants", ".kt", ".py", ".c", ".h", ".cpp")
                ):
                    self.file_map[file] = Path(root) / file
        print(f"Indexed {len(self.file_map)} source files.")

    def get_node_source(self, node_id: str, context_padding: int = 0) -> Optional[str]:
        if not self.code_dir:
            return None

        meta = self.node_metadata.get(node_id)
        if not meta or not (loc := meta.get("location")):
            return None

        target_uri = loc.uri or meta.get("file_uri")
        if not target_uri or not (source_file := self._resolve_file_path(target_uri)):
            return None

        try:
            with open(source_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                start = max(0, loc.startLine - 1 - context_padding)
                end = min(len(lines), loc.endLine + context_padding)
                return "".join(lines[start:end]) if start < len(lines) else None
        except Exception as e:
            print(f"Error reading source for {node_id}: {e}")
            return None

    def _resolve_file_path(self, uri: str) -> Optional[Path]:
        clean_path = uri.replace("file://", "")

        # 1. Direct match
        if (candidate := self.code_dir / clean_path).exists():
            return candidate

        # 2. Index match by filename
        filename = Path(clean_path).name
        if filename in self.file_map:
            candidate = self.file_map[filename]
            if str(candidate).replace("\\", "/").endswith(clean_path):
                return candidate
        return None

    def _build_index(self):
        cache_path = self.data_dir / ".embeddings_cache.pt"

        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}...")
            try:
                cache = torch.load(cache_path, weights_only=False)
                self.embeddings = cache["embeddings"]
                self.node_ids_list = cache["node_ids"]
                print(f"Loaded {len(self.node_ids_list)} cached embeddings.")
                return
            except Exception as e:
                print(f"Cache load failed: {e}, recomputing...")

        print(f"Computing embeddings for {len(self.node_metadata)} nodes...")
        texts, self.node_ids_list = [], []

        for node_id, data in self.node_metadata.items():
            display = data.get("display_name") or node_id
            kind = data.get("kind", "UNKNOWN")

            source = self.get_node_source(node_id)
            code_snippet = source[:500].replace("\n", " ") if source else ""
            texts.append(f"Type: {kind}; Name: {display}; Code: {code_snippet}")
            self.node_ids_list.append(node_id)

        if texts and self.model:
            batch_size = 128 if self.device == "cuda" else 32
            self.embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            print("Embeddings computed.")

            print(f"Caching embeddings to {cache_path}...")
            torch.save(
                {
                    "embeddings": self.embeddings,
                    "node_ids": self.node_ids_list,
                },
                cache_path,
            )
            print("Cache saved.")

    def find_nodes(self, query: str, limit: int = 5) -> List[Dict]:
        if self.embeddings is not None and self.model:
            query_emb = self.model.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_emb, self.embeddings)[0]
            top_results = torch.topk(cos_scores, k=min(limit, len(self.node_ids_list)))

            return [
                {
                    "id": self.node_ids_list[idx],
                    "score": float(score),
                    **self.node_metadata[self.node_ids_list[idx]],
                }
                for score, idx in zip(top_results.values, top_results.indices)
            ]

        # Fallback keyword search
        matches = []
        q_lower = query.lower()
        for nid, data in self.node_metadata.items():
            score = 0
            if q_lower in (data.get("display_name") or "").lower():
                score += 2
            if q_lower in nid.lower():
                score += 1
            if score > 0:
                matches.append((score, nid, data))

        matches.sort(key=lambda x: x[0], reverse=True)
        return [{"id": m[1], **m[2]} for m in matches[:limit]]

    def get_context_subgraph(
        self, node_ids: List[str], hops: int = 1
    ) -> Dict[str, Any]:
        nodes = set(node_ids)
        current = set(node_ids)

        for _ in range(hops):
            nxt = set()
            for nid in current:
                if nid in self.graph:
                    nxt.update(self.graph.successors(nid))
                    nxt.update(self.graph.predecessors(nid))
            nodes.update(nxt)
            current = nxt

        sub = self.graph.subgraph(nodes)
        return {
            "nodes": [
                {"id": n, **self.node_metadata[n]}
                for n in nodes
                if n in self.node_metadata
            ],
            "edges": [
                {"source": u, "target": v, "type": d.get("type", "unknown")}
                for u, v, d in sub.edges(data=True)
            ],
        }

    def format_context_for_llm(
        self, subgraph: Dict[str, Any], code_context_padding: int = 0
    ) -> str:
        output = ["CODEBASE CONTEXT:\n\nEntities:"]
        nodes_by_kind = {}
        for node in subgraph["nodes"]:
            nodes_by_kind.setdefault(node["kind"], []).append(node)

        for kind, nodes in nodes_by_kind.items():
            output.append(f"  [{kind}]")
            for node in nodes:
                name = node.get("display_name") or node["id"]
                output.append(f"    - {name} (ID: {node['id']})")

                if src := self.get_node_source(
                    node["id"], context_padding=code_context_padding
                ):
                    block = "\n".join(f"        {line}" for line in src.splitlines())
                    loc = node["location"]
                    output.append(
                        f"      Code Location: {loc.uri} L{loc.startLine}-L{loc.endLine}"
                    )
                    if code_context_padding:
                        output.append(
                            f"      (Context: +/- {code_context_padding} lines)"
                        )
                    output.append(f"      Source:\n{block}\n")

        output.append("\nRelationships:")
        for edge in subgraph["edges"]:
            s = self.node_metadata.get(edge["source"], {}).get(
                "display_name", edge["source"]
            )
            t = self.node_metadata.get(edge["target"], {}).get(
                "display_name", edge["target"]
            )
            output.append(f"  - {s} --[{edge['type']}]--> {t}")

        return "\n".join(output)


def demo():
    data_path = Path("data/glide/.semanticgraphs")
    if not data_path.exists():
        print(f"Data path {data_path} not found.")
        return

    rag = GraphRAG(data_dir=data_path.parent, code_dir="code/glide-4.5.0")
    print("\n--- Demo Query: 'Cache' ---")

    if nodes := rag.find_nodes("Cache", limit=2):
        print(f"Found {len(nodes)} starting nodes.")
        subgraph = rag.get_context_subgraph([n["id"] for n in nodes], hops=1)
        print(
            f"Subgraph: {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges."
        )
        print("\nGenerated LLM Context:")
        print("-" * 40)
        print(rag.format_context_for_llm(subgraph, code_context_padding=5))
        print("-" * 40)
    else:
        print("No nodes found for 'Cache'")


if __name__ == "__main__":
    demo()
