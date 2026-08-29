(in-package #:llm-backend-vllm-cpp)

(defvar *chat-fn* #'vllm-cpp:chat
  "Injected for tests. (lambda (engine request-json) response-json).")

(defvar *chat-stream-fn* #'vllm-cpp:chat-stream
  "Injected for tests. (lambda (engine request-json on-delta)).")

(defun %env (name)
  (let ((v (uiop:getenv name)))
    (and v (plusp (length v)) v)))

(defclass vllm-cpp-backend (llm-backend)
  ((model-path :initarg :model-path :accessor vllm-cpp-model-path :initform nil)
   (device :initarg :device :accessor vllm-cpp-device :initform :auto)
   (engine :initarg :engine :accessor vllm-cpp-engine :initform nil)))

(defun make-vllm-cpp-backend (&key model-path device engine)
  (make-instance 'vllm-cpp-backend
                 :model-path (or model-path (%env "VLLM_MODEL_PATH")
                                 (%env "VLLM_CPP_MODEL"))
                 :device (or device (vllm-cpp:default-device))
                 :engine engine))

(defun use-vllm-cpp-backend (&rest args &key &allow-other-keys)
  (setf *llm-backend* (apply #'make-vllm-cpp-backend args)))

(defun close-vllm-cpp-backend (backend)
  (when (vllm-cpp-engine backend)
    (vllm-cpp:free-engine (vllm-cpp-engine backend))
    (setf (vllm-cpp-engine backend) nil))
  backend)

(defun ensure-vllm-cpp-engine (backend)
  (or (vllm-cpp-engine backend)
      (setf (vllm-cpp-engine backend)
            (vllm-cpp:load-engine :model-path (vllm-cpp-model-path backend)
                                  :device (vllm-cpp-device backend)))))

(defmethod backend-model ((backend vllm-cpp-backend))
  (or (vllm-cpp-model-path backend)
      (and (vllm-cpp-engine backend)
           (vllm-cpp:engine-model-path (vllm-cpp-engine backend)))))

(defmethod backend-supports-p ((backend vllm-cpp-backend) (feature (eql :tools)))
  t)

(defmethod backend-supports-p ((backend vllm-cpp-backend)
                               (feature (eql :structured-output)))
  t)

(defmethod backend-supports-p ((backend vllm-cpp-backend) (feature (eql :stream)))
  t)

(defmethod backend-supports-p ((backend vllm-cpp-backend) (feature (eql :thinking)))
  t)

(defmethod backend-supports-p ((backend vllm-cpp-backend) (feature (eql :responses)))
  nil)

(defun %ht (&rest kvs)
  (let ((h (make-hash-table :test 'equal)))
    (loop for (k v) on kvs by #'cddr
          unless (or (null k) (eq v :omit) (null v))
            do (setf (gethash k h) v))
    h))

(defun %wire-part (part)
  (etypecase part
    (llm-text-part (%ht "type" "text" "text" (or (llm-text-part-text part) "")))
    (llm-image-part
     (%ht "type" "image_url"
          "image_url" (%ht "url" (or (llm-image-part-url part)
                                     (and (llm-image-part-data part)
                                          (format nil "data:~a;base64,~a"
                                                  (or (llm-image-part-media-type part)
                                                      "image/png")
                                                  (llm-image-part-data part)))))))
    (llm-thinking-part nil)
    (llm-part nil)))

(defun %wire-tool-call (part)
  (%ht "id" (or (llm-tool-call-part-id part) "call_0")
       "type" "function"
       "function" (%ht "name" (llm-tool-call-part-name part)
                       "arguments"
                       (let ((a (llm-tool-call-part-arguments part)))
                         (if (stringp a) a (stack-json:encode a))))))

(defun %wire-turn (turn)
  (let* ((turn (coerce-turn turn))
         (role (string-downcase (symbol-name (llm-turn-role turn))))
         (texts (remove nil (mapcar #'%wire-part (llm-turn-parts turn))))
         (calls (remove-if-not #'llm-tool-call-part-p (llm-turn-parts turn)))
         (results (remove-if-not #'llm-tool-result-part-p (llm-turn-parts turn)))
         (thinking (find-if #'llm-thinking-part-p (llm-turn-parts turn))))
    (cond
      ((eq (llm-turn-role turn) :tool)
       (let ((r (or (first results)
                    (make-llm-tool-result-part :id nil :content (turn-text turn)))))
         (%ht "role" "tool"
              "tool_call_id" (llm-tool-result-part-id r)
              "name" (llm-tool-result-part-name r)
              "content" (or (llm-tool-result-part-content r) ""))))
      (t
       (let ((content (cond
                        ((and texts (null (rest texts))
                              (equal (gethash "type" (first texts)) "text")
                              (null calls))
                         (gethash "text" (first texts)))
                        (texts (map 'vector #'identity texts))
                        (t ""))))
         (let ((h (%ht "role" role "content" content)))
           (when calls
             (setf (gethash "tool_calls" h)
                   (map 'vector #'%wire-tool-call calls)))
           (when thinking
             (setf (gethash "reasoning_content" h) (llm-thinking-part-text thinking)))
           h))))))

(defun %wire-tool (tool)
  (cond
    ((llm-tool-p tool)
     (%ht "type" "function"
          "function" (%ht "name" (llm-tool-name tool)
                          "description" (llm-tool-description tool)
                          "parameters" (or (llm-tool-parameters tool)
                                           (%ht "type" "object"
                                                "properties" (%ht))))))
    ((hash-table-p tool) tool)
    ((and (consp tool) (keywordp (car tool)))
     (%wire-tool (make-llm-tool :name (getf tool :name)
                                :description (getf tool :description)
                                :parameters (getf tool :parameters))))
    (t (error 'llm-error :message (format nil "not a tool: ~s" tool)))))

(defun %wire-tool-choice (choice)
  (etypecase choice
    (null nil)
    ((eql :auto) "auto")
    ((eql :none) "none")
    ((eql :required) "required")
    (string (%ht "type" "function" "function" (%ht "name" choice)))
    (hash-table choice)))

(defun %extra (settings)
  (and settings (llm-settings-extra settings)))

(defun %extra-get (extra key)
  (cond
    ((null extra) nil)
    ((hash-table-p extra) (or (gethash key extra)
                              (gethash (string-downcase (symbol-name key)) extra)))
    ((and (consp extra) (keywordp (car extra))) (getf extra key))
    (t nil)))

(defun %schema-name (schema)
  (cond
    ((symbolp schema) (string-downcase (symbol-name schema)))
    ((and (hash-table-p schema) (gethash "title" schema))
     (princ-to-string (gethash "title" schema)))
    (t "output")))

(defun %response-format (settings)
  (or (and settings (llm-settings-response-format settings))
      (let ((out (and settings (llm-settings-output settings))))
        (when out
          (%ht "type" "json_schema"
               "json_schema"
               (%ht "name" (%schema-name out)
                    "strict" t
                    "schema" (structured-output-json-schema out)))))))

(defun %chat-body (backend turns &key model settings tools tool-choice)
  (let* ((settings (coerce-settings settings))
         (extra (%extra settings))
         (body (%ht "model" (or model (backend-model backend) "vllm")
                    "messages" (map 'vector #'%wire-turn (coerce-turns turns))
                    "temperature" (and settings (llm-settings-temperature settings))
                    "max_tokens" (and settings (llm-settings-max-tokens settings))
                    "stop" (and settings (llm-settings-stop settings))
                    "top_p" (and settings (llm-settings-top-p settings))
                    "top_k" (%extra-get extra :top-k)
                    "min_p" (%extra-get extra :min-p)
                    "repetition_penalty" (%extra-get extra :repetition-penalty)
                    "min_tokens" (%extra-get extra :min-tokens)
                    "response_format" (%response-format settings)
                    "tools" (and tools (map 'vector #'%wire-tool
                                            (llm-protocol::%as-list tools)))
                    "tool_choice" (%wire-tool-choice tool-choice))))
    (let ((guided (or (%extra-get extra :guided-json)
                      (and settings (llm-settings-output settings)
                           (ignore-errors
                            (structured-output-json-schema
                             (llm-settings-output settings)))))))
      (when (and guided (not (gethash "response_format" body)))
        (setf (gethash "guided_json" body)
              (if (stringp guided) guided (stack-json:encode guided)))))
    (dolist (pair '((:guided-regex . "guided_regex")
                    (:guided-grammar . "guided_grammar")))
      (let ((v (%extra-get extra (car pair))))
        (when v (setf (gethash (cdr pair) body) v))))
    body))

(defun %finish-reason (raw)
  (cond
    ((or (null raw) (eq raw :null)) :stop)
    ((string-equal raw "stop") :stop)
    ((string-equal raw "length") :length)
    ((or (string-equal raw "tool_calls") (string-equal raw "tool_use")) :tool-use)
    ((string-equal raw "content_filter") :content-filter)
    ((string-equal raw "abort") :stop)
    (t :stop)))

(defun %usage (obj)
  (when (hash-table-p obj)
    (make-llm-usage
     :input-tokens (or (gethash "prompt_tokens" obj) (gethash "input_tokens" obj))
     :output-tokens (or (gethash "completion_tokens" obj) (gethash "output_tokens" obj))
     :total-tokens (gethash "total_tokens" obj))))

(defun %str (x)
  (cond
    ((or (null x) (eq x :null)) "")
    ((stringp x) x)
    (t (princ-to-string x))))

(defun %parse-chat (json requested-model)
  (let* ((obj (if (hash-table-p json) json (stack-json:decode json)))
         (choice (let ((cs (and (hash-table-p obj) (gethash "choices" obj))))
                   (and cs (plusp (length cs)) (elt cs 0))))
         (msg (and choice (gethash "message" choice)))
         (content (and msg (gethash "content" msg)))
         (tcs (and msg (gethash "tool_calls" msg)))
         (thinking (and msg (or (gethash "reasoning_content" msg)
                                (gethash "reasoning" msg)
                                (gethash "thinking" msg))))
         (parts (append
                 (and thinking (not (eq thinking :null))
                      (list (make-llm-thinking-part :text (%str thinking))))
                 (and content (not (eq content :null)) (plusp (length (%str content)))
                      (list (make-llm-text-part :text (%str content))))
                 (mapcar #'llm-protocol::%coerce-tool-call-part
                         (llm-protocol::%as-list tcs)))))
    (make-llm-response
     :parts parts
     :model (or (and (hash-table-p obj) (gethash "model" obj)) requested-model)
     :finish-reason (%finish-reason (and choice (gethash "finish_reason" choice)))
     :usage (%usage (and (hash-table-p obj) (gethash "usage" obj))))))

(defmethod generate ((backend vllm-cpp-backend) turns &key model settings
                     tools tool-choice output)
  (declare (ignore output))
  (let* ((engine (ensure-vllm-cpp-engine backend))
         (body (%chat-body backend turns :model model :settings settings
                           :tools tools :tool-choice tool-choice))
         (json (funcall *chat-fn* engine (stack-json:encode body))))
    (%parse-chat json (or model (backend-model backend)))))

(defmethod stream-generate ((backend vllm-cpp-backend) turns &key model
                            settings tools tool-choice on-part output)
  (declare (ignore output))
  (let* ((engine (ensure-vllm-cpp-engine backend))
         (body (%chat-body backend turns :model model :settings settings
                           :tools tools :tool-choice tool-choice))
         (acc (make-array 0 :element-type 'character :adjustable t :fill-pointer 0)))
    (funcall *chat-stream-fn* engine (stack-json:encode body)
             (lambda (delta finished)
               (declare (ignore finished))
               (when (and delta (plusp (length delta)))
                 (let* ((chunk (ignore-errors (stack-json:decode delta)))
                        (choice (and (hash-table-p chunk)
                                     (let ((cs (gethash "choices" chunk)))
                                       (and cs (plusp (length cs)) (elt cs 0)))))
                        (d (and choice (gethash "delta" choice)))
                        (text (and (hash-table-p d) (gethash "content" d))))
                   (cond
                     ((and text (stringp text) (plusp (length text)))
                      (loop for c across text do (vector-push-extend c acc))
                      (when on-part
                        (funcall on-part (make-llm-text-part :text text))))
                     ((and (null chunk) (plusp (length delta)))
                      (loop for c across delta do (vector-push-extend c acc))
                      (when on-part
                        (funcall on-part (make-llm-text-part :text delta)))))))
               t))
    (make-llm-response
     :parts (list (make-llm-text-part :text (copy-seq acc)))
     :model (or model (backend-model backend))
     :finish-reason :stop)))

(defmethod list-models ((backend vllm-cpp-backend) &key)
  (list (make-llm-model-info
         :id (or (backend-model backend) "vllm")
         :owned-by "vllm.cpp")))
