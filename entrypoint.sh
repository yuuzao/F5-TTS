#!/bin/bash

uv run uvicorn f5_tts.apiserver:app --host 0.0.0.0 --port 38100 /proc/1/fd/1 2>/proc/1/fd/2 &
uv run "f5-tts_infer-gradio" --host 0.0.0.0 --port 38101 >/proc/1/fd/1 2>/proc/1/fd/2 &
