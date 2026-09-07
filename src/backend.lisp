(in-package #:llm-backend-vllm-cpp)

(defvar *chat-fn* #'vllm-cpp:chat
  "Injected for tests. (lambda (engine request-json) response-json).")

(defvar *chat-stream-fn* #'vllm-cpp:chat-stream
  "Injected for tests. (lambda (engine request-json on-delta)).")

(defvar *embed-fn* #'vllm-cpp:embed
  "Injected for tests. (lambda (engine texts) → (values vectors dim prompt-tokens)).")

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

(defmethod backend-supports-p ((backend vllm-cpp-backend) (feature (eql :embeddings)))
  t)

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

(defun %ensure-tool-acc (table index)
  (or (gethash index table)
      (setf (gethash index table) (list :id nil :name nil :arguments ""))))

(defun %apply-tool-delta (table tc)
  (when (hash-table-p tc)
    (let* ((idx (or (gethash "index" tc) 0))
           (acc (%ensure-tool-acc table idx))
           (fn (or (gethash "function" tc) tc)))
      (when (gethash "id" tc)
        (setf (getf acc :id) (gethash "id" tc)))
      (when (and (hash-table-p fn) (gethash "name" fn))
        (setf (getf acc :name) (gethash "name" fn)))
      (let ((args (and (hash-table-p fn) (gethash "arguments" fn))))
        (when (and args (not (eq args :null)))
          (setf (getf acc :arguments)
                (concatenate 'string (getf acc :arguments) (%str args)))))
      (setf (gethash idx table) acc))))

(defun %tool-acc-parts (table)
  (mapcar (lambda (k)
            (let ((acc (gethash k table)))
              (make-llm-tool-call-part
               :id (getf acc :id)
               :name (or (getf acc :name) "")
               :arguments (or (getf acc :arguments) ""))))
          (sort (loop for k being the hash-keys of table collect k) #'<)))

(defmethod stream-generate ((backend vllm-cpp-backend) turns &key model
                            settings tools tool-choice on-part output)
  (declare (ignore output))
  (let* ((engine (ensure-vllm-cpp-engine backend))
         (body (%chat-body backend turns :model model :settings settings
                           :tools tools :tool-choice tool-choice))
         (text (make-string-output-stream))
         (thinking (make-string-output-stream))
         (tool-acc (make-hash-table :test 'eql))
         (finish :stop)
         (usage nil)
         (seen-model nil))
    (funcall *chat-stream-fn* engine (stack-json:encode body)
             (lambda (delta finished)
               (declare (ignore finished))
               (when (and delta (plusp (length delta)))
                 (let ((chunk (ignore-errors (stack-json:decode delta))))
                   (cond
                     ((hash-table-p chunk)
                      (when (gethash "model" chunk)
                        (setf seen-model (gethash "model" chunk)))
                      (when (gethash "usage" chunk)
                        (setf usage (%usage (gethash "usage" chunk))))
                      (let* ((choice (let ((cs (gethash "choices" chunk)))
                                       (and cs (plusp (length cs)) (elt cs 0))))
                             (d (and choice (gethash "delta" choice)))
                             (reason (and choice (gethash "finish_reason" choice))))
                        (when (and reason (not (eq reason :null)))
                          (setf finish (%finish-reason reason)))
                        (when (hash-table-p d)
                          (let ((content (gethash "content" d))
                                (think (or (gethash "reasoning_content" d)
                                           (gethash "reasoning" d)
                                           (gethash "thinking" d)))
                                (tcs (gethash "tool_calls" d)))
                            (when (and content (not (eq content :null))
                                       (plusp (length (%str content))))
                              (write-string (%str content) text)
                              (when on-part
                                (funcall on-part (make-llm-text-part
                                                  :text (%str content)))))
                            (when (and think (not (eq think :null))
                                       (plusp (length (%str think))))
                              (write-string (%str think) thinking)
                              (when on-part
                                (funcall on-part (make-llm-thinking-part
                                                  :text (%str think)))))
                            (dolist (tc (llm-protocol::%as-list tcs))
                              (%apply-tool-delta tool-acc tc))))))
                     (t
                      (write-string delta text)
                      (when on-part
                        (funcall on-part (make-llm-text-part :text delta)))))))
               t))
    (let* ((think-s (get-output-stream-string thinking))
           (text-s (get-output-stream-string text))
           (calls (%tool-acc-parts tool-acc)))
      (dolist (call calls)
        (when on-part (funcall on-part call)))
      (make-llm-response
       :parts (append (and (plusp (length think-s))
                           (list (make-llm-thinking-part :text think-s)))
                      (and (plusp (length text-s))
                           (list (make-llm-text-part :text text-s)))
                      calls)
       :model (or seen-model model (backend-model backend))
       :finish-reason (if calls :tool-use (or finish :stop))
       :usage usage))))

(defmethod list-models ((backend vllm-cpp-backend) &key)
  (list (make-llm-model-info
         :id (or (backend-model backend) "vllm")
         :owned-by "vllm.cpp")))

(defun %slice-embedding (vec dimensions)
  (cond
    ((null dimensions) vec)
    ((> dimensions (length vec))
     (error 'llm-error
            :message (format nil "requested dimensions ~a > model dim ~a"
                             dimensions (length vec))))
    (t (subseq vec 0 dimensions))))

(defmethod embed ((backend vllm-cpp-backend) inputs &key model dimensions
                  encoding-format)
  (when (and encoding-format
             (not (member encoding-format '(:float "float") :test #'equal)))
    (error 'llm-unsupported
           :message (format nil "vllm.cpp embeddings are float-only, got ~s"
                            encoding-format)))
  (let* ((texts (coerce-embed-inputs inputs))
         (engine (ensure-vllm-cpp-engine backend)))
    (multiple-value-bind (vecs dim tokens)
        (funcall *embed-fn* engine texts)
      (declare (ignore dim))
      (make-llm-embed-result
       :embeddings (loop for v in vecs for i from 0
                         collect (make-llm-embedding
                                  :vector (%slice-embedding v dimensions)
                                  :index i))
       :model (or model (backend-model backend))
       :usage (make-llm-usage :input-tokens tokens :total-tokens tokens)))))
