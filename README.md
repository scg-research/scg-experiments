# Quick start

```bash
docker-compose up -d              # Start Neo4j
uv run python -m src.bootstrap    # Run full pipeline

npx @modelcontextprotocol/inspector uv run python -m src.mcp_server # Start MCP Inspector
```

# Queries

```sql
MATCH (n:CodeNode)-[r:CALL|TYPE]->(m:CodeNode)
WHERE n.id CONTAINS "DiskCache"
RETURN n, r, m LIMIT 50
```

#### Explicit nodes

```sql
MATCH (n:CodeNode) WHERE n.kind IS NOT NULL RETURN n LIMIT 5
```

#### Implicit nodes

```sql
MATCH (n:Node) WHERE n.kind IS NULL RETURN n LIMIT 5
```

#### Count nodes

```sql
MATCH (n) RETURN count(n)
```

#### Show nodes and direct relationships

```sql
MATCH (n)
WITH n LIMIT 200
MATCH (n)-[r]-(m)
RETURN n, r, m
```

# MCP Inspector 
```bash
npx @modelcontextprotocol/inspector uv run python -m src.mcp_server
```