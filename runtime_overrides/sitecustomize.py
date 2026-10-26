import os
if os.environ.get("CIR_STORAGE_RETRY_ENABLED") == "1":
    try:
        from cir_storage_retry import install
        install()
    except Exception:
        os._exit(87)
