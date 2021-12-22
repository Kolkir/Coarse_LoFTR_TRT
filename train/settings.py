class TrainSettings:
    def __init__(self):
        self.cuda = True

        # first 25 epochs
        # self.batch_size = 2
        # self.batch_size_divider = 1  # Used for gradient accumulation
        # self.use_amp = False

        # other with AMP
        self.batch_size = 32
        self.batch_size_divider = 8  # Used for gradient accumulation
        self.use_amp = True

        self.learning_rate = 0.01
        self.optimizer_beta1 = 0.9
        self.optimizer_beta2 = 0.999
        self.optimizer_eps = 1.0e-8
        self.optimizer_weight_decay = 0.01
        self.epochs = 100

        self.with_teacher = False
        self.temperature = 10.0

        self.amp_scale = 2.**8
        self.data_loader_num_workers = 4
        self.write_statistics = True
        self.statistics_period = 10

        self.image_size = (640, 480)
