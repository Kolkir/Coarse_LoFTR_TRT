from src.magicpointtrainer import MagicPointTrainer
from src.superpointtrainer import SuperPointTrainer
from src.superpoint import SuperPoint


class TrainWrapper(object):
    def __init__(self, checkpoint_path, settings):
        self.checkpoint_path = checkpoint_path
        self.settings = settings
        self.net = SuperPoint(self.settings)
        if settings.cuda:
            self.net = self.net.cuda()
            print('Model moved to GPU')
        self.net.train()

    def train_magic_point(self, synthetic_dataset_path, use_coco=False):
        self.net.disable_descriptor()
        magic_point_trainer = MagicPointTrainer(synthetic_dataset_path, self.checkpoint_path, self.settings, use_coco)
        magic_point_trainer.train('magic_point', self.net)

    def train_super_point(self, coco_dataset_path, magic_point_weights):
        super_point_trainer = SuperPointTrainer(coco_dataset_path, self.checkpoint_path, magic_point_weights,
                                                self.settings)
        super_point_trainer.train('super_point', self.net)
