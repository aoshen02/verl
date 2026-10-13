# First build the pinned checkout's docker/Dockerfile.uv.cu130 as verl:uv-cu130.
ARG VERL_BASE_IMAGE=verl:uv-cu130
FROM ${VERL_BASE_IMAGE}
WORKDIR /workspace/verl
# A fixed, image-owned environment avoids per-node installs and shared-venv races.
ENV UV_PROJECT_ENVIRONMENT=/opt/verl-venv
RUN uv sync --frozen --all-packages --extra vllm --extra fsdp --no-dev
RUN uv pip install --python /opt/verl-venv/bin/python 'reasoning-gym==0.1.25'
ENV VIRTUAL_ENV=/opt/verl-venv
ENV PATH="/opt/verl-venv/bin:${PATH}"
RUN python -c 'import importlib.metadata as m; assert m.version("vllm")=="0.29.0"; assert m.version("transformers")=="5.12.1"; import flash_attn, fla, transfer_queue'
RUN sha256sum uv.lock > /opt/verl-uv-lock.sha256
CMD ["bash"]
