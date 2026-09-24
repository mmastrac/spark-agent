# Model recipe checklist

Every model repo (`spark-<name>`) is checked against this list.

## Layout

```
model.yaml
.env.example
README.md
KNOBS.md
.gitmodules
.submodules/spark-agent/
compose/
image/
smoketest/
```

## model.yaml

- [ ] `name` matches the served model name and the mentat group
- [ ] `huggingface` id of the exact checkpoint served
- [ ] `description`: what it is and why this model was chosen
- [ ] `vram.budget_gib`, `vram.resident_gib` (measured, with the date), `vram.kv_pin_bytes`
- [ ] `tp`, `pp`
- [ ] `api` and `endpoints`
- [ ] Nothing in it duplicates a value the entrypoint also sets

## .env.example

- [ ] Every variable in `compose/` that has no default, listed
- [ ] Per-node values called out as per-node
- [ ] Placeholders only: no fleet addresses, box names or secrets
- [ ] `.env` is in `.gitignore`

## mentat integration

- [ ] TP=1 / PP=1: `python -m ray.register &` beside the engine, no `ray` executable
- [ ] Multi-node: `ray start` from the mentatd binary, plus `mentatd-probe-machine`
- [ ] `MENTAT_GROUP` defaults to the served name
- [ ] `MENTAT_OPENAI_API` in port form (`8000/v1`), on the API rank only
- [ ] `MENTAT_MCP_API` in port form, on every rank
- [ ] `MENTAT_MODEL_PROVIDER` set (`vllm`)
- [ ] `MENTAT_NODE_IP` unset unless the stack is single-node on loopback
- [ ] Shim version matches the fleet daemons (`MENTAT_VERSION`)
- [ ] Shows as healthy in mentat-serve's `/`, and a request through the router answers
- [ ] Status-server tools show in the router's `/mcp`

## .submodules/spark-agent

- [ ] Submodule at `.submodules/spark-agent`, HTTPS URL
- [ ] `vllm -> .submodules/spark-agent/vllm` symlink
- [ ] No local copy of `status-server.py`
- [ ] No `AGENT_URL` or `/register`: the agent has no registry
- [ ] Pin bumped on purpose, never left behind on a stale commit

## compose/

- [ ] `<name>.yaml`, run with `docker compose -f compose/<name>.yaml up -d`
- [ ] Every variable is `${VAR:-default}` or `${VAR:?why}`
- [ ] No fleet addresses, box names or host paths as defaults
- [ ] `network_mode: host`; ports disjoint from the other tenants on the box
- [ ] No added capabilities: host tools belong to the agent
- [ ] KV pinned (`--kv-cache-memory` or equivalent), never derived from profiling
- [ ] Model weights mounted read-only
- [ ] JIT caches (FlashInfer, TileLang, Triton, torch) on a persistent mount
- [ ] Restart policy set

## image/

- [ ] Everything the Dockerfile copies lives here: Dockerfile, entrypoint, patches, verify script
- [ ] Build context is the repo root (`docker build -f image/Dockerfile .`), so the `vllm` symlink stays inside it
- [ ] Base pinned by digest or nightly SHA, never `latest` or `nightly`
- [ ] `ARG BASE` and `ARG MENTAT_VERSION` above the first `FROM`
- [ ] `mentat-artifacts` pulled registry-qualified (`mmastrac/...`)
- [ ] Shim wheel installed `--no-deps`
- [ ] `verify-base.py` fails the build if the model arch, the sm_120 cubins or `ray.register` are missing
- [ ] Entrypoint baked in, not bind-mounted
- [ ] Entrypoint reads its knobs from the environment; `KNOBS.md` regenerated
- [ ] `build.sh` with a `TAG`; builds from a plain `rsync -a` copy

## smoketest/

- [ ] `run.sh <base> [served-name]`, exit code is the failure count
- [ ] `lib.sh` byte-identical to the other repos
- [ ] One `t_*` per case, every check a jq expression
- [ ] Fails against the wrong model, not only passes against the right one

## Docs

- [ ] README: what it serves, where, how to run it, how to roll back
- [ ] Measured numbers carry a date
- [ ] No fleet addresses or box names
