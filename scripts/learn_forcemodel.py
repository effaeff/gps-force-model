"""Main script for running training procedure"""

import shutil
import misc
import numpy as np

from pytorchutils.mlp import MLPModel
from gpsforcemodel.forcemodel import ForceModel
from pytorchutils.globals import nn
from gpsforcemodel.dataprocessor import DataProcessor
from gpsforcemodel.trainer import Trainer

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from config import (
    data_config,
    model_config,
    OPT,
    DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    NB_MODELS,
    DILATION_BASE,
    KERNEL_SIZE,
    WINDOW
)

def objective(config):
    """Wrapper for training runs"""
    if OPT:
        if WINDOW:
            dilate = config['dilate']
            n_window = int(config['n_window'])
            print(f"Config: {config}")
            if dilate:
                nb_layers = int(
                    np.ceil(
                        (
                            np.log(
                                ((n_window - 1) * (DILATION_BASE - 1)) / (KERNEL_SIZE - 1) + 1
                            ) / np.log(DILATION_BASE)
                        )
                    )
                )
                dilation = [DILATION_BASE**idx for idx in range(nb_layers)]
            else:
                nb_layers = int(np.ceil((n_window - 1) / (KERNEL_SIZE - 1)))
                dilation = [1 for __ in range(nb_layers)]

            channels = [2**(5 + idx) for idx in range(nb_layers)]
            padding = [d * (KERNEL_SIZE - 1) for d in dilation]

        data_config['n_window'] = int(config['n_window'])
        data_config['batch_size'] = int(config['batch_size'])
        model_config['n_window'] = n_window
        model_config['learning_rate'] = config['lr']
        model_config['reg_lambda'] = config['reg_lambda']
        model_config['batch_size'] = int(config['batch_size'])
        model_config['nb_layers'] = nb_layers
        model_config['dilation'] = dilation
        model_config['channels'] = channels
        model_config['padding'] = padding

    data_processor = DataProcessor(data_config)
    model = nn.DataParallel(ForceModel(model_config))
    # model = ForceModel(model_config)
    # model = [nn.DataParallel(ForceModel(model_config)) for __ in range(NB_MODELS)]

    trainer = Trainer(model_config, model, data_processor)
    get_batches_fn = data_processor.get_next_batches
    trainer.get_batches_fn = get_batches_fn

    # trainer.validate(-1, True, True)
    if OPT:
        acc = trainer.train(validate_every=1, save_every=0, save_eval=False, verbose=False)
    else:
        acc = trainer.train(validate_every=1, save_every=1, save_eval=True, verbose=True)

    print(f"Config: {config}, acc: {acc:.2f}")

    return {'loss': acc, 'params': config, 'status': STATUS_OK}

def main():
    """Main method"""
    misc.gen_dirs([DATA_DIR, RESULTS_DIR, MODELS_DIR])

    config = {
        'batch_size': 1024,
        'dilate': True,
        'lr': 0.0005943494747246402,
        'n_window': 7,
        'reg_lambda': 0.008552688977854465
    }
    if not OPT:
        objective(config)
        quit()

    search_space = {
        'n_window': hp.randint('n_window', 3, 8),
        'dilate': hp.choice('dilate', [True, False]),
        'lr': hp.uniform("lr", 0.00001, 0.001),
        'reg_lambda': hp.uniform('reg_lambda', 0.0001, 0.01),
        'batch_size': hp.randint("batch_size", 64, 2049)
    }

    trials = Trials()
    best = fmin(objective, space=search_space, algo=tpe.suggest, max_evals=75, trials=trials)
    np.save(f"{RESULTS_DIR}/hyperopt_best.npy", best)
    print(f"Finished hyperparameter tuning. Best config:\n{best}")

if __name__ == "__main__":
    main()
