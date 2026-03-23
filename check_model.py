import os
import argparse
import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool

import mxnet as mx


def find_prefix_and_epoch(model_dir: str):
    files = os.listdir(model_dir)
    symbol_files = [f for f in files if f.endswith("-symbol.json")]
    param_files = [f for f in files if f.endswith(".params")]

    if not symbol_files:
        raise FileNotFoundError("Không thấy *-symbol.json")
    if not param_files:
        raise FileNotFoundError("Không thấy *.params")

    sym = symbol_files[0]
    prefix = sym[:-len("-symbol.json")]

    matched = [f for f in param_files if f.startswith(prefix + "-")]
    if not matched:
        raise RuntimeError("Không có params khớp với symbol")

    # ví dụ model-0000.params -> epoch = 0
    p = matched[0]
    epoch = int(p.replace(prefix + "-", "").replace(".params", ""))
    return os.path.join(model_dir, prefix), epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    prefix, epoch = find_prefix_and_epoch(args.model_dir)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    print("=== SYMBOL OUTPUTS ===")
    print(sym.list_outputs())

    print("\n=== ARG PARAMS ===")
    for k, v in sorted(arg_params.items()):
        print(f"{k:40s} {tuple(v.shape)}")

    print("\n=== AUX PARAMS ===")
    for k, v in sorted(aux_params.items()):
        print(f"{k:40s} {tuple(v.shape)}")

    print("\n=== CHECK fc1 RELATED ===")
    for dname, d in [("arg", arg_params), ("aux", aux_params)]:
        for k, v in sorted(d.items()):
            if "fc1" in k.lower():
                print(f"[{dname}] {k:40s} {tuple(v.shape)}")


if __name__ == "__main__":
    main()