# Batch loop / VRAM cleanup

`config/loop.env` controls the sequential model loop.

Important: 5 minutes = 300 seconds. 1800 seconds is 30 minutes.

The batch runner performs this sequence for each model:

1. Run the benchmark.
2. Verify the expected per-model Markdown report exists and is non-empty.
3. Request `keep_alive=0` during Ollama generation when enabled.
4. Wait for the configured cleanup delay.
5. Run `ollama stop <model>`.
6. Poll `ollama ps` until the model disappears.
7. Optionally poll `nvidia-smi` until GPU memory is below the configured threshold.
8. Wait `TASK_LOOP_WAIT_SECONDS` before starting the next model.

A full NVIDIA GPU reset (`nvidia-smi --gpu-reset`) is intentionally NOT used by default. On laptops/desktops it can disrupt the display driver or other CUDA applications. Releasing the Ollama model via `keep_alive=0` + `ollama stop` is the safer approach.
