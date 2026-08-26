(defsystem "llm-protocol-vllm-cpp"
  :version "0.1.0"
  :description "llm-protocol backend over vllm-cpp (mudler/vllm.cpp C ABI)"
  :author "egao1980"
  :license "MIT"
  :depends-on ("llm-protocol" "vllm-cpp" "json-protocol" "json-backend-jzon")
  :properties
  (:cl-repo
   (:ci (:with ("llm-protocol/capability"))))
  :serial t
  :pathname "src"
  :components ((:file "package")
               (:file "backend"))
  :in-order-to ((test-op (test-op "llm-protocol-vllm-cpp/tests"))))

(defsystem "llm-protocol-vllm-cpp/tests"
  :depends-on ("llm-protocol-vllm-cpp" "llm-protocol/capability" "rove")
  :pathname "tests"
  :serial t
  :components ((:file "package")
               (:file "backend-test"))
  :perform (test-op (o c)
             (unless (symbol-call :rove :run c)
               (error "tests failed for ~A" (component-name c)))))
