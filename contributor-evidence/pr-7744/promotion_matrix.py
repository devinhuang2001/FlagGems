"""Compare the real generic eq entrypoints with the native CUDA torch oracle."""

import itertools
import json
import sys
import traceback

import torch

from flag_gems.ops.eq import eq, eq_scalar


def main():
    dtypes = [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64,
              torch.float16, torch.float32, torch.float64]
    scalars = [False, True, -1, 0, 1, 2, 1.25, -1.5, 2**24 + 1,
               float(2**24 + 1), 2**53 + 1, float(2**53 + 1)]
    checks = []
    failures = []

    def check(name, candidate, reference):
        try:
            actual = candidate()
            expected = reference()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            checks.append(name)
        except Exception:
            failures.append({"case": name, "traceback": traceback.format_exc()})

    for strided in [False, True]:
        values = {}
        for dtype in dtypes:
            data = [0, 1, -1, 2, 3] if not dtype.is_floating_point else [0, 1.25, -1.5, 2.5, 3]
            tensor = torch.tensor(data, dtype=dtype, device="cuda")
            if strided:
                storage = torch.empty(10, dtype=dtype, device="cuda")
                storage[::2].copy_(tensor)
                tensor = storage[::2]
            values[dtype] = tensor
        for left, right in itertools.product(dtypes, repeat=2):
            x, y = values[left][:, None], values[right][None, :]
            check(f"tensor:{left}:{right}:strided={strided}",
                  lambda: eq(x, y), lambda: torch.eq(x, y))
        for dtype, scalar in itertools.product(dtypes, scalars):
            x = values[dtype]
            check(f"scalar:{dtype}:{scalar!r}:strided={strided}",
                  lambda: eq_scalar(x, scalar), lambda: torch.eq(x, scalar))

    for dtype, base in [(torch.int32, 2**24), (torch.int64, 2**24), (torch.int64, 2**53)]:
        x = torch.tensor([base, base + 1, base + 2], dtype=dtype, device="cuda")
        for scalar in [base + 1, float(base + 1)]:
            check(f"boundary:{dtype}:{base}:scalar={scalar!r}",
                  lambda: eq_scalar(x, scalar), lambda: torch.eq(x, scalar))

    torch.cuda.synchronize()
    result = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "passed": len(checks), "failed": len(failures), "failures": failures}
    print(json.dumps(result, indent=2))
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
