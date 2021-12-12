class TrainSettings:
    def __init__(self):
        self.cuda = True 

        self.batch_size = 4
        self.batch_size_divider = 1  # Used for gradient accumulation

        self.learning_rate = 0.001
        self.optimizer_beta1 = 0.9
        self.optimizer_beta2 = 0.999
        self.optimizer_eps = 1.0e-8
        self.optimizer_weight_decay = 0.01
        self.epochs = 100

        self.temperature = 10

        self.use_amp = True
        self.data_loader_num_workers = 4
        self.write_statistics = True
        self.statistics_period = 10

        self.image_size = (640, 480)
