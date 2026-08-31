import torch


M = 4
N = 4
K = 4

if not torch.cuda.is_available():
    raise RuntimeError("Part II requires a CUDA GPU.")

torch.manual_seed(663)
torch.set_printoptions(precision=8, sci_mode=False)

matrix_a = torch.rand((M, N), dtype=torch.float32, device="cuda")
matrix_b = torch.rand((N, K), dtype=torch.float32, device="cuda")

print("PART II - FLOATING-POINT PRECISION")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"A shape: {tuple(matrix_a.shape)}, B shape: {tuple(matrix_b.shape)}\n")

for scale in (1, 100):
    a = matrix_a * scale
    b = matrix_b * scale

    result_float32 = a @ b
    result_float16 = a.to(torch.float16) @ b.to(torch.float16)
    result_bfloat16 = a.to(torch.bfloat16) @ b.to(torch.bfloat16)

    float16_error = (result_float32 - result_float16.float()).abs().max().item()
    bfloat16_error = (result_float32 - result_bfloat16.float()).abs().max().item()

    print(f"--- Input scale = {scale} ---")
    print("float32 result:")
    print(result_float32.cpu())

    print("\nfloat16 result:")
    print(result_float16.float().cpu())
    print(f"Maximum absolute difference from float32: {float16_error:.8g}")

    print("\nbfloat16 result:")
    print(result_bfloat16.float().cpu())
    print(f"Maximum absolute difference from float32: {bfloat16_error:.8g}")
    print()


