import os
import configparser
import logging

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
config_database_file = os.path.join(CONFIG_DIR, "antifraud.cfg")


class Config(object):
    # Parameters for attention model
    def __init__(self, filepath=config_database_file):
        self.__config = configparser.ConfigParser()
        self.__config.read(filepath)
        self.input_shape_2d = (9, 9)
        self.input_shape_3d = (64, 8, 7)
        self.num_classes = 2
        self.filter_sizes = [2,3,4]
        self.num_filters = [6,12,24]
        self.attention_hidden_dim = 100
        self.batch_size = 256
        self.num_epochs = 16
        self.evaluate_every = 25
        self.test_size = 0.3

    def get_config(self):
        return self.__config

    @staticmethod
    def __get_log_level(levels):
        return {
            'logging.INFO': logging.INFO,
            'logging.DEBUG': logging.DEBUG,
            'logging.WARNING': logging.WARNING,
            'logging.ERROR': logging.ERROR,
        }[levels]

    def get_log_level(self):
        return self.__get_log_level(self.__config["log"]["log.level"])
