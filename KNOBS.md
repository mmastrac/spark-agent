# Knobs

The agent has no entrypoint script. It reads these directly
(`agent/spark-agent.py`). The compose file requires `LOG_DIR` (the host side
of `/logs`), `MENTAT_ROUTER_URL` and `ALLOWED_SOURCES` in the node's `.env`.

| variable | default | meaning |
|---|---|---|
| `AGENT_LOG_DIR` | `/logs` | log and snapshot directory, mounted read-only |
| `MENTAT_ROUTER_URL` | (none) | mentat-serve, e.g. `http://router:6381`; source of the model table |
| `MENTAT_GROUP` | `agent-<hostname>` | group the agent registers its MCP under |
| `MENTAT_DAEMON` | `127.0.0.1:6379` | local mentatd control address |
| `ALLOWED_SOURCES` | `127.0.0.1` | address prefixes let in; the router's host is always let in |
| `LOAD_INTERVAL_S` | `5` | how often the router, peers and engines are read |
| `AGENT_PORT` | `8090` | port peers are linked on when `--port` is not given |
| `PY_SPY` | `py-spy` | py-spy binary for `engine_stacks` |
| `SPARK_MEMORY_PY` | `/usr/local/bin/spark-memory.py` | helper behind `memory_accounting` |
