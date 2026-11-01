FROM python:3.12.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir -e .
COPY assets ./assets
COPY config ./config

ENTRYPOINT ["python", "-m", "so101_dice_sim"]

FROM runtime AS test
RUN python -m pip install --no-cache-dir -e ".[test]"
COPY tests ./tests
ENTRYPOINT ["python", "-m", "pytest"]

FROM runtime AS visualization
RUN apt-get update \
    && apt-get install -y --no-install-recommends libosmesa6 ffmpeg fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir -e ".[visualization]"
ENV MUJOCO_GL=osmesa \
    MPLBACKEND=Agg
ENTRYPOINT ["python", "-m", "so101_dice_sim.visualization"]

FROM visualization AS visualization-test
RUN python -m pip install --no-cache-dir -e ".[test]"
COPY tests ./tests
ENTRYPOINT ["python", "-m", "pytest", "tests/test_visualization.py"]

FROM visualization AS learning
ENV HF_HOME=/app/outputs/learning/cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HUB_DISABLE_XET=1 \
    WANDB_MODE=disabled \
    DO_NOT_TRACK=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*
COPY config/learning-requirements.txt /tmp/learning-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/learning-requirements.txt \
    && python -m pip check
ENTRYPOINT ["python", "-m", "so101_dice_sim.learning"]
