# llm-protocol-vllm-cpp

[`llm-protocol`](https://github.com/egao1980/llm-protocol) backend over native [`vllm-cpp`](https://github.com/egao1980/vllm-cpp) (`mudler/vllm.cpp` C ABI). Not HTTP.

`generate` / `stream-generate` → `vllm_chat` / `vllm_chat_stream`. `respond` falls back to `items->turns` then `generate` (no `/responses` wire).

The published **linux/amd64** `vllm-cpp` overlay **is CUDA**. `make-vllm-cpp-backend` defaults `:device` to `vllm-cpp:default-device` (`:cuda` on linux/amd64, `:auto` elsewhere). Override with `:device` or `VLLM_DEVICE`. CPU linux and MLX are local `vllm-cpp` flavors, not extra OCI platforms.

```lisp
(asdf:load-system "llm-protocol-vllm-cpp")
(let ((b (stack-llm-vllm-cpp:make-vllm-cpp-backend
          :model-path (uiop:getenv "VLLM_MODEL_PATH"))))
  (stack-llm:llm-response-text
   (stack-llm:generate b "ping" :settings '(:temperature 0 :max-tokens 32))))
```

`VLLM_MODEL_PATH` fills an omitted `:model-path`. Live: `VLLM_CPP_LIVE=1`. CI needs published `llm-protocol` + `vllm-cpp` on GHCR (`vllm-cpp` without an overlay still loads; generate skips).

Part of [cl-stack](https://github.com/egao1980/cl-stack) ([#195](https://github.com/egao1980/cl-stack/issues/195)).

## License

MIT — see [LICENSE](LICENSE).
