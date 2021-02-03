import tensorflow as tf


class globalConfig:
    def __init__(self):
        super(globalConfig, self).__init__()

        self.batch_size = 256
        self.epochs = 80
        self.learning_rate = 1e-4


class modelConfig:
    def __init__(self):
        super(modelConfig, self).__init__()

        self.data_size = (28, 28)
        self.latent_size = 100
        self.drop_rate = 0.3
