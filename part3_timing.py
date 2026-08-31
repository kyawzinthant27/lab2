import time

import torch


SIZE = 2048
REPEATS = 3
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


if not torch.cuda.is_available():
    raise RuntimeError("Part III requires a CUDA GPU.")

torch.set_num_threads(4)

print("PART III - COMPUTATION TIME")
print(f"Matrix multiplication: ({SIZE}, {SIZE}) @ ({SIZE}, {SIZE})")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Repeated measurements: {REPEATS}\n")

for device_name in ("cuda", "cpu"):
    print(f"--- {device_name.upper()} ---")

    for dtype in DTYPES:
        matrix_a = torch.rand((SIZE, SIZE), dtype=dtype, device=device_name)
        matrix_b = torch.rand((SIZE, SIZE), dtype=dtype, device=device_name)

        result = matrix_a @ matrix_b
        if device_name == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(REPEATS):
            result = matrix_a @ matrix_b

        if device_name == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        average_ms = ((end - start) / REPEATS) * 1000

        print(f"{str(dtype).replace('torch.', '')}: {average_ms:.3f} ms")

    print()

