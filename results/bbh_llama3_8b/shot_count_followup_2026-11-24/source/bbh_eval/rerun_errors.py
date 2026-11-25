"""Distinguish an actual CUDA allocation failure from unrelated worker errors."""
import sys
import traceback


CUDA_OOM_EXIT = 86
CUDA_OOM_MARKER = "BBH_WORKER_EXIT_CUDA_OOM"


def worker_main(main):
    try:
        main()
    except RuntimeError as err:
        # torch.cuda.OutOfMemoryError is an alias of torch.OutOfMemoryError in
        # several versions. Verify its CUDA message to avoid classifying CPU OOM.
        torch = sys.modules.get("torch")
        oom_class = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", ())
        supported = torch is not None and (type(err) is RuntimeError or isinstance(err, oom_class))
        gpu_oom = str(err).strip().lower().startswith(("cuda out of memory", "cuda error: out of memory"))
        if not (supported and gpu_oom):
            raise
        traceback.print_exc()
        print(CUDA_OOM_MARKER, file=sys.stderr, flush=True)
        raise SystemExit(CUDA_OOM_EXIT) from None
