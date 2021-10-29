import argparse
from loftr import LoFTR, default_cfg
import torch
import trtorch


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
        missed_keys, unexpected_keys = matcher.load_state_dict(checkpoint['state_dict'])
        if len(missed_keys) > 0:
            print('Checkpoint is broken')
            return 1
        print('Successfully loaded pre-trained weights.')
    else:
        print('Failed to load checkpoint')
        return 1
    matcher = matcher.eval().to(device=device)

    with torch.no_grad():
        enabled_precisions = {torch.float32}
        trtorch.logging.set_reportable_log_level(trtorch.logging.Level.Debug)

        # export_backbone(enabled_precisions, matcher)

        # export_loftr_coarse(enabled_precisions, matcher)

        # TODO: doesn't work on master TRTorch - problem with dynamic shape
        # export_loftr_fine(enabled_precisions, matcher)

        # TODO: doesn't work on GTX 1060 - indexes should have type `long`
        # export_fine_preprocess(enabled_precisions, matcher)


def export_fine_preprocess(enabled_precisions, matcher):
    fine_preprocess_script = torch.jit.script(matcher.fine_preprocess)
    print('fine preprocess model scripted', flush=True)
    input_shape1 = [1, 128, 240, 320]
    input_shape2 = [1, 4800, 256]
    coarse_inputs = [trtorch.Input(shape=input_shape1, dtype=torch.float32),
                     trtorch.Input(shape=input_shape1, dtype=torch.float32),
                     trtorch.Input(shape=input_shape2, dtype=torch.float32),
                     trtorch.Input(shape=input_shape2, dtype=torch.float32),
                     trtorch.Input(shape=(4800,), dtype=torch.int32),
                     trtorch.Input(shape=(4800,), dtype=torch.int32),
                     trtorch.Input(shape=(4800,), dtype=torch.int32)]
    fine_preprocess_script.eval()
    trt_ts_module = trtorch.compile(fine_preprocess_script,
                                    inputs=coarse_inputs,
                                    workspace_size=1 << 32,
                                    enabled_precisions=enabled_precisions,
                                    strict_types=False,
                                    disable_tf32=True,
                                    debug=True
                                    )
    torch.jit.save(trt_ts_module, "trt_fine_preprocess.pt")
    test = torch.jit.load("trt_fine_preprocess.pt")
    print('TRTorch  fine preprocess exported', flush=True)


def export_loftr_fine(enabled_precisions, matcher):
    loftr_fine_script = torch.jit.script(matcher.loftr_fine)
    print('loftr fine model scripted', flush=True)
    # input shape can have a different size
    min_shape = (1, 25, 128)
    opt_shape = (2400, 25, 128)
    max_shape = (4800, 25, 128)

    fine_inputs = [trtorch.Input(min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape, dtype=torch.float32),
                   trtorch.Input(min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape, dtype=torch.float32)]
    loftr_fine_script.eval()
    trt_ts_module = trtorch.compile(loftr_fine_script,
                                    inputs=fine_inputs,
                                    workspace_size=1 << 32,
                                    enabled_precisions=enabled_precisions,
                                    strict_types=False,
                                    disable_tf32=True,
                                    debug=True
                                    )
    torch.jit.save(trt_ts_module, "trt_loftr_fine.pt")
    test = torch.jit.load("trt_loftr_fine.pt")
    print('TRTorch  loftr fine exported', flush=True)


def export_loftr_coarse(enabled_precisions, matcher):
    loftr_coarse_script = torch.jit.script(matcher.loftr_coarse)
    print('loftr coarse model scripted', flush=True)
    input_shape = [1, 4800, 256]
    coarse_inputs = [trtorch.Input(shape=input_shape, dtype=torch.float32),
                     trtorch.Input(shape=input_shape, dtype=torch.float32)]
    loftr_coarse_script.eval()
    trt_ts_module = trtorch.compile(loftr_coarse_script,
                                    inputs=coarse_inputs,
                                    workspace_size=1 << 32,
                                    enabled_precisions=enabled_precisions,
                                    strict_types=False,
                                    disable_tf32=True,
                                    debug=True
                                    )
    torch.jit.save(trt_ts_module, "trt_loftr_coarse.pt")
    test = torch.jit.load("trt_loftr_coarse.pt")
    print('TRTorch  loftr coarse exported', flush=True)


def export_backbone(enabled_precisions, matcher):
    backbone_script = torch.jit.script(matcher.backbone)
    print('backbone model scripted', flush=True)
    input_shape = [2, 1, 480, 640]
    backbone_inputs = [trtorch.Input(shape=input_shape, dtype=torch.float32)]
    backbone_script.eval()
    trt_ts_module = trtorch.compile(backbone_script,
                                    inputs=backbone_inputs,
                                    workspace_size=1 << 32,
                                    enabled_precisions=enabled_precisions,
                                    strict_types=False,
                                    disable_tf32=True,
                                    debug=True
                                    )
    torch.jit.save(trt_ts_module, "trt_loftr_backbone.pt")
    test = torch.jit.load("trt_loftr_backbone.pt")
    print('TRTorch backbone exported', flush=True)


if __name__ == "__main__":
    main()
