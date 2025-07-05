# syntax=docker.io/docker/dockerfile:1.7-labs
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Create user and install dependencies
RUN useradd -l -m -s /bin/bash -u ${NON_ROOT_UID} ${NON_ROOT_USER} && \
    apt update && \
    apt -y install curl git libenchant-2-2 && \
    apt clean

# Environment variables
ENV PYTHONIOENCODING utf8
ENV LANG "C.UTF-8"
ENV LC_ALL "C.UTF-8"
ENV PATH "${HOME_DIR}/.local/bin:${PATH}"
ENV UV_LINK_MODE=copy
USER ${NON_ROOT_USER}
WORKDIR ${HOME_DIR}

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin

# Copy files
COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} ${REPO_DIR}/pyproject.toml ${HOME_DIR}/direct-marketing-optimization/pyproject.toml
COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} ${REPO_DIR}/src ${HOME_DIR}/direct-marketing-optimization/src

WORKDIR /${HOME_DIR}/direct-marketing-optimization
COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} --exclude=src ${REPO_DIR} ${HOME_DIR}/direct-marketing-optimization

# docker build -f docker/direct-marketing.Dockerfile .