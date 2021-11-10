import argparse
from loftr import LoFTR, default_cfg
import torch


def main():
    parser = argparse.ArgumentParser(description='LoFTR demo.')
    parser.add_argument('--weights', type=str, default='weights/outdoor_ds.ckpt',
                        help='Path to network weights.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cpu or cuda')

    opt = parser.parse_args()
    print(opt)

    device = torch.device(opt.device)

    print('Loading pre-trained network...')
    matcher = LoFTR(config=default_cfg)
    checkpoint = torch.load(opt.weights)
    if checkpoint is not None:
        missed_keys, unexpected_keys = matcher.load_state_dict(checkpoint['state_dict'], strict=False)
        if len(missed_keys) > 0:
            print('Checkpoint is broken')
            return 1
        print('Successfully loaded pre-trained weights.')
    else:
        print('Failed to load checkpoint')
        return 1
    matcher = matcher.eval().to(device=device)

    with torch.no_grad():
        dummy_image = torch.randn(1, 1, default_cfg['input_height'], default_cfg['input_width'], device=device)
        torch.onnx.export(matcher, (dummy_image, dummy_image), 'loftr.onnx', verbose=True, opset_version=11)


if __name__ == "__main__":
    main()
