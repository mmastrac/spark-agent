# The host agent for a GB10 node. Holds the privileged tools so the container
# running the model does not have to.
#
# Small on purpose: stdlib Python plus py-spy. nvidia-smi is not installed
# here, the NVIDIA container runtime injects it along with the driver
# libraries when the compose file reserves a GPU.
# Pulled registry-qualified: a bare `mentat-artifacts:<ver>` tag resolves to a
# local image, and those have been mislabelled before.
ARG MENTAT_VERSION=0.12.0
FROM mmastrac/mentat-artifacts:${MENTAT_VERSION} AS mentat
FROM python:3.12-slim

# pciutils/usbutils carry pci.ids and usb.ids, without which lspci and lsusb
# report bare hex vendor:device pairs. iproute2 backs the network tool.
RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 pciutils usbutils procps \
    && rm -rf /var/lib/apt/lists/* \
    && command -v ip && command -v lspci && command -v lsusb && command -v ps

# py-spy reads another process's Python stack without stopping it. Container
# processes are visible through the host's PID namespace, which the compose
# file shares, so this reaches the engine without the model container holding
# any tracing capability itself.
RUN pip install --no-cache-dir py-spy && py-spy --version

# The accounting helper runs live rather than reading the periodic snapshot,
# because pid: host lets this container see every process and nvidia-smi gives
# the GPU figure that process RSS never shows. The host's own memory timer
# runs the same script.
COPY spark-memory.py /usr/local/bin/spark-memory.py

# The mentat shim, for ray.register only: the agent announces its MCP endpoint
# to the local mentatd, and mentat-serve merges it. --no-deps, because the
# shim installs as `ray` and must not pull real ray in.
COPY --from=mentat /out/ /tmp/mentat-out/
RUN pip install --no-cache-dir --no-deps /tmp/mentat-out/mentatd-*-py3-none-any.whl \
    && rm -rf /tmp/mentat-out \
    && python3 -c "from ray import register; register.connect, register.parse_args"

COPY spark-agent.py /usr/local/bin/spark-agent.py
RUN chmod +x /usr/local/bin/spark-agent.py /usr/local/bin/spark-memory.py \
    && python3 -c "import ast; ast.parse(open('/usr/local/bin/spark-agent.py').read()); \
ast.parse(open('/usr/local/bin/spark-memory.py').read())"

# 8090 = MCP and the node status endpoint. The compose file decides which
# addresses it is reachable on.
EXPOSE 8090
ENTRYPOINT ["python3", "/usr/local/bin/spark-agent.py"]
CMD ["--port", "8090"]
