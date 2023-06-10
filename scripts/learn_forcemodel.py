"""Main script for running training procedure"""

import shutil
import misc

from pytorchutils.mlp import MLPModel
from gpsforcemodel.forcemodel import ForceModel
from pytorchutils.globals import nn
from gpsforcemodel.dataprocessor import DataProcessor
from gpsforcemodel.trainer import Trainer

from config import (
    data_config,
    model_config,
    DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    NB_MODELS
)

def main():
    """Main method()"""
    misc.gen_dirs([DATA_DIR, RESULTS_DIR, MODELS_DIR])

    data_processor = DataProcessor(data_config)
    model = nn.DataParallel(ForceModel(model_config))
    # model = [nn.DataParallel(ForceModel(model_config)) for __ in range(NB_MODELS)]

    trainer = Trainer(model_config, model, data_processor)
    get_batches_fn = data_processor.get_next_batches
    trainer.get_batches_fn = get_batches_fn

    # trainer.validate(-1, True)
    trainer.train(validate_every=1, save_every=1)

if __name__ == "__main__":
    main()
