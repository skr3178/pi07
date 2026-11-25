"""List tensor names/shapes in a torch .pt (zip) checkpoint without loading weights or running pickled code."""
import collections
import math
import pickle
import sys
import zipfile


class TensorStub:
    def __init__(self, dtype, key, offset, shape):
        self.dtype, self.shape = dtype, tuple(shape)
        self.region = (key, offset, math.prod(self.shape))  # same region = alias of the same bytes


class StorageType:
    def __init__(self, name):
        self.name = name


def rebuild_tensor(storage, offset, size, stride, *rest):
    return TensorStub(storage[0], storage[1], offset, size)


class SafeUnpickler(pickle.Unpickler):
    """Only allow the handful of globals a plain state dict needs; refuse everything else."""

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return StorageType(name)
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch" and name in ("bfloat16", "float16", "float32"):
            return name
        raise pickle.UnpicklingError(f"refused global {module}.{name}")

    def persistent_load(self, pid):
        # pid = ('storage', storage_type, key, location, numel)
        return (pid[1].name if isinstance(pid[1], StorageType) else str(pid[1]), pid[2])


bytes_per = {"BFloat16Storage": 2, "HalfStorage": 2, "FloatStorage": 4, "DoubleStorage": 8,
             "LongStorage": 8, "IntStorage": 4, "BoolStorage": 1, "ByteStorage": 1}

with zipfile.ZipFile(sys.argv[1]) as z:
    pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
    obj = SafeUnpickler(z.open(pkl)).load()

print("top-level type:", type(obj).__name__)
if isinstance(obj, dict):
    for k, v in obj.items():
        print(f"  key {k!r}: {type(v).__name__}" + (f" with {len(v)} entries" if isinstance(v, dict) else f" = {v!r}"[:120]))

tensors = {}


def walk(o, prefix=""):
    if isinstance(o, TensorStub):
        tensors[prefix] = o
    elif isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{prefix}.{k}" if prefix else str(k))


walk(obj)
# Count each storage region once: a checkpoint can register one tensor under several module paths.
seen, aliases = set(), 0
groups = collections.defaultdict(lambda: [0, 0, set()])
for name, t in tensors.items():
    if t.region in seen:
        aliases += 1
        continue
    seen.add(t.region)
    parts = name.split(".")
    key = ".".join(parts[:3]) if len(parts) > 3 else name
    n = math.prod(t.shape)
    groups[key][0] += n
    groups[key][1] += n * bytes_per.get(t.dtype, 0)
    groups[key][2].add(t.dtype)
total_n = sum(g[0] for g in groups.values())
total_b = sum(g[1] for g in groups.values())
print(f"\n{len(tensors)} tensor names, {aliases} of them aliases of another name; "
      f"{total_n / 1e9:.3f}B unique params, {total_b / 1e9:.2f} GB")
for k, (n, b, d) in sorted(groups.items(), key=lambda x: -x[1][0])[:25]:
    print(f"  {k:60s} {n / 1e6:9.1f}M  {b / 1e9:6.2f} GB  {sorted(d)}")
