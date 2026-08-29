(in-package #:llm-backend-vllm-cpp/tests)

(defun %ht (&rest kvs)
  (let ((h (make-hash-table :test 'equal)))
    (loop for (k v) on kvs by #'cddr
          unless (null v)
            do (setf (gethash k h) v))
    h))

(defun %fake-chat (engine request-json)
  (declare (ignore engine))
  (let* ((body (stack-json:decode request-json))
         (msgs (gethash "messages" body))
         (last (elt msgs (1- (length msgs))))
         (tools (gethash "tools" body)))
    (stack-json:encode
     (%ht "model" (or (gethash "model" body) "local")
          "usage" (%ht "prompt_tokens" 3 "completion_tokens" 2 "total_tokens" 5)
          "choices"
          (vector (%ht "finish_reason" (if tools "tool_calls" "stop")
                       "message"
                       (%ht "role" "assistant"
                            "content" (if tools
                                          :null
                                          (format nil "ok:~a" (gethash "content" last)))
                            "tool_calls"
                            (when tools
                              (vector (%ht "id" "call_1"
                                           "type" "function"
                                           "function"
                                           (%ht "name" "sum"
                                                "arguments" "{\"a\":1}")))))))))))

(defun %fake-stream (engine request-json on-delta)
  (declare (ignore engine request-json))
  (funcall on-delta
           (stack-json:encode
            (%ht "choices" (vector (%ht "delta" (%ht "content" "ok")))))
           nil)
  (funcall on-delta "" t))

(defmacro %with-fake (&body body)
  `(let ((llm-backend-vllm-cpp:*chat-fn* #'%fake-chat)
         (llm-backend-vllm-cpp:*chat-stream-fn* #'%fake-stream))
     ,@body))

(defun %backend ()
  (llm-backend-vllm-cpp:make-vllm-cpp-backend
   :model-path "/models/fake"
   :engine :fake))

(deftest generate-mock
  (%with-fake
    (let* ((b (%backend))
           (r (llm-protocol:generate b "hi" :model "local")))
      (ok (equal "ok:hi" (llm-protocol:llm-response-text r)))
      (ok (equal "local" (llm-protocol:llm-response-model r)))
      (ok (eq :stop (llm-protocol:llm-response-finish-reason r)))
      (ok (= 5 (llm-protocol:llm-usage-total-tokens
                (llm-protocol:llm-response-usage r)))))))

(deftest settings-on-wire
  (let ((seen nil))
    (%with-fake
      (let ((llm-backend-vllm-cpp:*chat-fn*
              (lambda (engine json)
                (declare (ignore engine))
                (setf seen (stack-json:decode json))
                (%fake-chat :fake json))))
        (llm-protocol:generate
         (%backend) "hi"
         :settings '(:temperature 0 :max-tokens 16 :extra (:top-k 8 :min-p 0.05)))
        (ok (zerop (gethash "temperature" seen)))
        (ok (= 16 (gethash "max_tokens" seen)))
        (ok (= 8 (gethash "top_k" seen)))
        (ok (< (abs (- (gethash "min_p" seen) 0.05)) 1e-6))))))

(deftest tools-mock
  (%with-fake
    (let ((r (llm-protocol:generate
              (%backend) "add"
              :tools (list (llm-protocol:make-llm-tool :name "sum")))))
      (ok (eq :tool-use (llm-protocol:llm-response-finish-reason r)))
      (ok (equal "sum" (llm-protocol:llm-tool-call-part-name
                        (first (llm-protocol:llm-response-tool-calls r))))))))

(deftest stream-generate-mock
  (%with-fake
    (let* ((parts nil)
           (r (llm-protocol:stream-generate
               (%backend) "hi"
               :on-part (lambda (p) (push p parts)))))
      (ok (equal "ok" (llm-protocol:llm-response-text r)))
      (ok (llm-protocol:llm-text-part-p (first parts))))))

(deftest respond-falls-back-to-generate
  (%with-fake
    (let ((r (llm-protocol:respond (%backend) "hi")))
      (ok (equal "ok:hi" (llm-protocol:llm-response-text r)))
      (ok (llm-protocol:llm-message-item-p
           (first (llm-protocol:llm-response-items r)))))))

(deftest list-models
  (let ((models (llm-protocol:list-models (%backend))))
    (ok (equal "/models/fake" (llm-protocol:llm-model-info-id (first models))))
    (ok (equal "vllm.cpp" (llm-protocol:llm-model-info-owned-by (first models))))))

(deftest default-device-follows-vllm-cpp
  (let ((b (llm-backend-vllm-cpp:make-vllm-cpp-backend :model-path "/models/fake")))
    (ok (eq (vllm-cpp:default-device)
            (llm-backend-vllm-cpp:vllm-cpp-device b)))))

(deftest catalogue
  (let ((cat (llm-protocol:make-llm-catalogue (%backend))))
    (ok (capability-protocol:capability-supported-p cat :llm-tools))
    (ok (capability-protocol:capability-supported-p cat :llm-structured-output))
    (ok (capability-protocol:capability-supported-p cat :llm-thinking))))

(deftest live-generate
  (let ((live (uiop:getenv "VLLM_CPP_LIVE"))
        (path (uiop:getenv "VLLM_MODEL_PATH")))
    (cond
      ((or (null live) (zerop (length live)))
       (skip "set VLLM_CPP_LIVE=1 for a live vllm.cpp call"))
      ((or (null path) (zerop (length path)))
       (skip "set VLLM_MODEL_PATH"))
      ((not (vllm-cpp:vllm-available-p))
       (skip "libvllm not present"))
      (t
       (let ((r (llm-protocol:generate
                 (llm-backend-vllm-cpp:make-vllm-cpp-backend)
                 "Reply with the single word pong and nothing else."
                 :settings '(:temperature 0 :max-tokens 32))))
         (ok (plusp (length (or (llm-protocol:llm-response-text r) "")))))))))
