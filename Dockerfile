FROM ghcr.io/astral-sh/uv:bookworm

RUN apt-get update && apt-get install -y vim git curl wget ffmpeg
WORKDIR /app
COPY . .

RUN uv venv --python 3.10
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -e .

RUN chmod +x entrypoint.sh

EXPOSE 38100 38101

CMD ["./entrypoint.sh"]
