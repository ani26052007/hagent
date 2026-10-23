# Sandbox image for the docker env backend. Needs bash, git, coreutils (timeout, sleep).
# Build: docker build -t hagent-sandbox examples/docker
# Add your project's own dependencies (Python packages, test tools) below.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m agent
