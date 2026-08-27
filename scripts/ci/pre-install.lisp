;;;; Stream-extract OCI layers. chipz:DECOMPRESS (buffer API) doubles the
;;;; output vector (2GiB → 4GiB) while still holding the gzip blob
;;;; (~1.2GiB for vllm-cpp linux/amd64 CUDA). 8GB SBCL heap still OOMs.
;;;; Stream gunzip → tar so peak is blob + one entry.

(in-package :cl-repository-client/installer)

(defun %stream-extract-tar-gz (tar-gz-data target-dir &key strip-prefix)
  (let* ((bytes (if (typep tar-gz-data '(simple-array (unsigned-byte 8) (*)))
                    tar-gz-data
                    (coerce tar-gz-data '(simple-array (unsigned-byte 8) (*)))))
         (input (flexi-streams:make-in-memory-input-stream bytes))
         (gz (chipz:make-decompressing-stream 'chipz:gzip input)))
    (extract-tar-stream gz target-dir :strip-prefix strip-prefix)))

(defun extract-layer-stripping-prefix (tar-gz-data target-dir prefix)
  (%stream-extract-tar-gz tar-gz-data target-dir :strip-prefix prefix))

(defun extract-layer (tar-gz-data target-dir)
  (%stream-extract-tar-gz tar-gz-data target-dir))

(format t "~&; ci: stream-extract OCI layers (chipz Gray stream)~%")
