from copy import deepcopy

from loftr import LoFTR
from loftr.utils.cvpr_ds_config import default_cfg
from utils import make_student_config


def print_model_size(model):
    param_num = 0
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_num += param.nelement()

    buffer_size = 0
    buffer_num = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_num += buffer.nelement()
    size_all_mb = (param_size + buffer_size) / 1024**2

    print("model parameters number: {:,}".format(param_num + buffer_num))
    print("model parameters size: {:.3f}MB".format(size_all_mb))


def main():
    teacher_cfg = deepcopy(default_cfg)
    student_cfg = make_student_config(default_cfg)

    teacher_model = LoFTR(config=teacher_cfg)
    print("Teacher model size:")
    print_model_size(teacher_model)

    student_model = LoFTR(config=student_cfg)
    print("Student model size:")
    print_model_size(student_model)


if __name__ == "__main__":
    main()
