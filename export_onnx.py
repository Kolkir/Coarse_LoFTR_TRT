import argparse

import onnx
import torch
import torch.nn.utils.prune as prune

from loftr import LoFTR, default_cfg
from utils import make_student_config


def main():
    parser = argparse.ArgumentParser(description="LoFTR demo.")
    parser.add_argument(
        "--out_file",
        type=str,
        default="weights/LoFTR_teacher.onnx",
        help="Path for the output ONNX model.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/LoFTR_teacher.pt",  # weights/outdoor_ds.ckpt
        help="Path to network weights.",
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="If specified the original LoFTR model will be used.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--prune", default=False, help="Do unstructured pruning")

    opt = parser.parse_args()
    print(opt)

    device = torch.device(opt.device)

    if opt.original:
        model_cfg = default_cfg
    else:
        model_cfg = make_student_config(default_cfg)

    print("Loading pre-trained network...")
    model = LoFTR(config=model_cfg)
    checkpoint = torch.load(opt.weights)
    if checkpoint is not None:
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint["model_state_dict"]
        missed_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missed_keys) > 0:
            print("Checkpoint is broken")
            return 1
        print("Successfully loaded pre-trained weights.")
    else:
        print("Failed to load checkpoint")
        return 1

    if opt.prune:
        print("Model pruning")
        for name, module in model.named_modules():
            # prune connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name="weight", amount=0.5)
                prune.remove(module, "weight")
            # prune connections in all linear layers
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=0.5)
                prune.remove(module, "weight")
        weight_total_sum = 0
        weight_total_num = 0
        for name, module in model.named_modules():
            # prune connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                weight_total_sum += torch.sum(module.weight == 0)
            # prune connections in all linear layers
            elif isinstance(module, torch.nn.Linear):
                weight_total_num += module.weight.nelement()

        print(f"Global sparsity: {100. * weight_total_sum / weight_total_num:.2f}")

    print(f"Moving model to device: {device}")
    model = model.eval().to(device=device)

    with torch.no_grad():
        dummy_image = torch.randn(
            1, 1, default_cfg["input_height"], default_cfg["input_width"], device=device
        )
        torch.onnx.export(
            model,
            (dummy_image, dummy_image),
            opt.out_file,
            verbose=True,
            opset_version=11,
        )

    model = onnx.load(opt.out_file)
    onnx.checker.check_model(model)


if __name__ == "__main__":
    main()
