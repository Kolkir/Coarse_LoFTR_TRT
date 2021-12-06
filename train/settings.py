class TrainSettings:
    def __init__(self):
        self.cuda = False

        self.batch_size = 1
        self.batch_size_divider = 1  # Used for gradient accumulation

        self.learning_rate = 0.001
        self.optimizer_beta1 = 0.9
        self.optimizer_beta2 = 0.999
        self.optimizer_eps = 1.0e-8
        self.optimizer_weight_decay = 0.01
        self.epochs = 100

        self.use_amp = False
        self.data_loader_num_workers = 0
        self.write_statistics = True
        self.statistics_period = 1

        self.image_size = (640, 480)
