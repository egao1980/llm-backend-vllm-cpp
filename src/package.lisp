(defpackage #:llm-protocol-vllm-cpp
  (:use #:cl #:llm-protocol)
  (:nicknames #:stack-llm-vllm-cpp)
  (:export #:vllm-cpp-backend
           #:make-vllm-cpp-backend
           #:use-vllm-cpp-backend
           #:vllm-cpp-model-path
           #:vllm-cpp-device
           #:vllm-cpp-engine
           #:ensure-vllm-cpp-engine
           #:close-vllm-cpp-backend
           #:*chat-fn*
           #:*chat-stream-fn*))

(in-package #:llm-protocol-vllm-cpp)
