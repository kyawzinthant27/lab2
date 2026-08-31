import torch


ROWS = 10_000
COLS = 10_000
MIB = 1024**2

if not torch.cuda.is_available():
    raise RuntimeError("Part I requires a CUDA GPU.")

print("PART I - MEMORY FOOTPRINT")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Matrix dimensions: {ROWS} x {COLS}\n")

for dtype in (torch.float32, torch.float16):
    torch.cuda.empty_cache()

    memory_before = torch.cuda.memory_allocated()

    matrix = torch.rand(
        (ROWS, COLS),
        dtype=dtype,
        device="cuda",
    )
    torch.cuda.synchronize()

    memory_after = torch.cuda.memory_allocated()
    matrix_memory = memory_after - memory_before

    print(f"{str(dtype).replace('torch.', '')}")
    print(f"Bytes per element: {matrix.element_size()}")
    print(f"GPU memory allocated: {matrix_memory / MIB:.2f} MiB\n")

    del matrix


