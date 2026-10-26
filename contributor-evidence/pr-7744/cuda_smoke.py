import json

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(X, Y, Z, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < N
    tl.store(Z + index, tl.load(X + index, mask, 0) + tl.load(Y + index, mask, 0), mask)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    x = torch.arange(1024, device="cuda", dtype=torch.int64)
    y = x + 2**24
    z = torch.empty_like(x)
    add_kernel[(4,)](x, y, z, x.numel(), BLOCK=256)
    torch.cuda.synchronize()
    torch.testing.assert_close(z, x + y, rtol=0, atol=0)
    print(json.dumps({
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "triton": triton.__version__, "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "int64_triton_add": "PASS",
    }, indent=2))
