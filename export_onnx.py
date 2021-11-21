import argparse
from loftr import LoFTR, default_cfg
import torch
import torch.nn.utils.prune as prune


def main():
    parser = argparse.ArgumentParser(description='LoFTR demo.')
    parser.add_argument('--weights', type=str, default='weights/outdoor_ds.ckpt',
                        help='Path to network weights.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cpu or cuda')
    parser.add_argument('--prune', default=False, help='Do unstructured pruning')

    opt = parser.parse_args()
    print(opt)

    device = torch.device(opt.device)

    print('Loading pre-trained network...')
    model = LoFTR(config=default_cfg)
    checkpoint = torch.load(opt.weights)
    if checkpoint is not None:
        missed_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
        if len(missed_keys) > 0:
            print('Checkpoint is broken')
            return 1
        print('Successfully loaded pre-trained weights.')
    else:
        print('Failed to load checkpoint')
        return 1

    if opt.prune:
        print('Model pruning')
        for name, module in model.named_modules():
            # prune connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=0.5)
                prune.remove(module, 'weight')
            # prune connections in all linear layers
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.5)
                prune.remove(module, 'weight')
        weight_total_sum = 0
        weight_total_num = 0
        for name, module in model.named_modules():
            # prune connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                weight_total_sum += torch.sum(module.weight == 0)
            # prune connections in all linear layers
            elif isinstance(module, torch.nn.Linear):
                weight_total_num += module.weight.nelement()

        print(f'Global sparsity: {100. * weight_total_sum / weight_total_num:.2f}')

    print(f'Moving model to device: {device}')
    model = model.eval().to(device=device)

    with torch.no_grad():
        dummy_image = torch.randn(1, 1, default_cfg['input_height'], default_cfg['input_width'], device=device)
        torch.onnx.export(model, (dummy_image, dummy_image), 'loftr.onnx', verbose=True, opset_version=11)


if __name__ == "__main__":
    main()
