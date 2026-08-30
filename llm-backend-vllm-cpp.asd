(defsystem "llm-backend-vllm-cpp"
  :version "0.3.0"
  :description "llm-protocol backend over vllm-cpp (mudler/vllm.cpp C ABI)"
  :author "egao1980"
  :license "MIT"
  :depends-on ("llm-protocol" "vllm-cpp" "json-protocol" "json-backend-jzon")
  :properties
  (:cl-repo
   ;; llm-protocol/capability is a secondary system in the llm-protocol
   ;; tarball, not an OCI package. SAT-missed transitive:
   (:ci (:with ("capability-protocol"))))
  :serial t
  :pathname "src"
  :components ((:file "package")
               (:file "backend"))
  :in-order-to ((test-op (test-op "llm-backend-vllm-cpp/tests"))))

(defsystem "llm-backend-vllm-cpp/tests"
  :depends-on ("llm-backend-vllm-cpp" "llm-protocol/capability" "rove")
  :pathname "tests"
  :serial t
  :components ((:file "package")
               (:file "backend-test"))
  :perform (test-op (o c)
             (unless (symbol-call :rove :run c)
               (error "tests failed for ~A" (component-name c)))))
