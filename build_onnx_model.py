#!/usr/bin/env python3
"""
Build a PWA ONNX model directly (no PyTorch dependency).

The graph is constructed node-by-node using the `onnx` library.
Complex numbers are split into (real, imag) float pairs throughout.
"""
import sys, os, argparse, numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ampfit.config_loader import Config
from ampfit._onnx_builder import PWAONNXBuilder

OP = onnx.helper.make_node


def float_type():
    return TensorProto.FLOAT


def int64():
    return TensorProto.INT64


def main():
    parser = argparse.ArgumentParser(description="Build PWA ONNX model")
    parser.add_argument("--config", default="config_angle.yml")
    parser.add_argument("--output", default="pwa_forward.onnx")
    parser.add_argument("--output-norm", default="pwa_forward_norm.onnx")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--no-norm", action="store_true",
                        help="Skip building the norm model")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = Config(args.config)
    kernel_config = config.build_all_index()
    builder = PWAONNXBuilder(kernel_config)

    # ── Forward model (NLL + gradients) ──
    print(f"Building forward ONNX graph (batch_size={args.batch_size})...")
    model = builder.build(batch_size=args.batch_size, norm_model=False)
    # Strip unused initializers to silence onnxruntime warnings
    used = set()
    for n in model.graph.node:
        used.update(n.input)
        used.update(n.output)
    kept = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.save(model, args.output)
    print(f"✓ Saved to {args.output}")
    print(f"  Inputs: {len(model.graph.input)}")
    print(f"  Outputs: {len(model.graph.output)}")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Constants: {len(model.graph.initializer)} (was {len(kept)+len(model.graph.initializer)})")

    if args.validate:
        print("\nValidating forward model...")
        import onnxruntime as ort
        sess = ort.InferenceSession(args.output)
        for i in sess.get_inputs():
            print(f"  Input {i.name}: {i.shape}")
        for o in sess.get_outputs():
            print(f"  Output {o.name}: {o.shape}")
        print("✓ Forward model loads successfully")

    # ── Norm model (sum(P*weight) + gradients) ──
    if not args.no_norm:
        print(f"\nBuilding norm ONNX graph (batch_size={args.batch_size})...")
        norm_model = builder.build(batch_size=args.batch_size, norm_model=True)
        used = set()
        for n in norm_model.graph.node:
            used.update(n.input); used.update(n.output)
        kept = [i for i in norm_model.graph.initializer if i.name in used]
        del norm_model.graph.initializer[:]
        norm_model.graph.initializer.extend(kept)
        onnx.save(norm_model, args.output_norm)
        print(f"✓ Saved to {args.output_norm}")
        print(f"  Inputs: {len(norm_model.graph.input)}")
        print(f"  Outputs: {len(norm_model.graph.output)}")
        print(f"  Nodes: {len(norm_model.graph.node)}")
        print(f"  Constants: {len(norm_model.graph.initializer)}")

        if args.validate:
            print("\nValidating norm model...")
            import onnxruntime as ort
            sess2 = ort.InferenceSession(args.output_norm)
            for i in sess2.get_inputs():
                print(f"  Input {i.name}: {i.shape}")
            for o in sess2.get_outputs():
                print(f"  Output {o.name}: {o.shape}")
            print("✓ Norm model loads successfully")


if __name__ == "__main__":
    main()
