import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import trtorch


class LeNetFeatExtractor(nn.Module):
    def __init__(self):
        super(LeNetFeatExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x


class LeNetClassifier(nn.Module):
    def __init__(self):
        super(LeNetClassifier, self).__init__()
        self.fc1 = nn.Linear(16 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feat = LeNetFeatExtractor()
        self.classifer = LeNetClassifier()

    def forward(self, x):
        x = self.feat(x)
        x = self.classifer(x)
        return x


model = LeNet()
model = model.cuda()
with torch.no_grad():
    model.eval()
    input = torch.rand([1, 1, 16, 16], dtype=torch.float32, device='cuda')
    output = model(input)
    script_model = torch.jit.script(model)
    script_model.eval()
    trtorch.logging.set_reportable_log_level(trtorch.logging.Level.Debug)
    compile_settings = {
        "inputs": [trtorch.Input(
                shape=[1, 1, 16, 16],
                dtype=torch.float32),
        ],
        "enabled_precisions": {torch.float32},
        "disable_tf32": True,
        "sparse_weights": True,
        "debug": True
    }
    trt_ts_module = trtorch.compile(script_model, **compile_settings)
