# spark-agent

The MCP side of the GB10 / ASUS GX10 Spark fleet, in two parts:

| path | what |
|---|---|
| `agent/` | The host agent image. One per node, privileged, serves no model. |
| `vllm/` | The status server that runs inside a vLLM model container. |

Both register with mentat. mentat-serve merges every group's MCP into one
endpoint, so clients reach every node and every model container through the
router, and `__group` picks which one answers.

## agent/

Holds the tools that need the host: every process, ptrace for py-spy, the
GPU, logs, and the host's snapshots. The model containers then need no added
capabilities.

    cd agent && TAG=spark-agent:v6 ./build.sh
    docker compose -f spark-agent.yaml up -d

It registers its MCP endpoint with the local mentatd as group
`agent-<hostname>`. Its page on `:8090` shows every node and every model,
read from mentat-serve (`MENTAT_ROUTER_URL`).

## vllm/

`status-server.py` serves a model container's stage page and its MCP tools.
A recipe takes it as a submodule rather than a copy:

    git submodule add git@github.com:mmastrac/spark-agent.git .submodules/spark-agent
    ln -s .submodules/spark-agent/vllm vllm

and its Dockerfile copies `vllm/status-server.py`. BuildKit follows a symlink
that stays inside the build context, so the image gets the file itself.
