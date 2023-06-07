"""Config file"""

RANDOM_SEED = 1234

DATA_DIR = 'data/aggreg5'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'
PARAMS = f'{DATA_DIR}/LHS_Final_new.xlsx'

INPUT_SIZE = 6
OUTPUT_SIZE = 3
TARGET_OUTPUT_SIZE = 3
STEPY = 75
EDGES = 2
BATCH_SIZE = 32
N_WINDOW = 16

K_C = 272.94663445441347
K_N = 344.1285160407239
K_T = -50.57065214836188
M_C = 0.4118070182339672
M_N = 0.27309617279279746
M_T = 0.3288884071807737

data_config = {
    'keys': ['t', 'fx', 'fy', 'fz'],
    'group': 'Recorder',
    'data_dir': DATA_DIR,
    'results_dir': RESULTS_DIR,
    'params': PARAMS,
    'excel_sheet': 'LHS_new',
    'scaler': ['x_scaler.scal', 'y_scaler.scal'],
    'test_size': 0.4, # Half of the test set is used for validation
    'negate': [-1, 1, 1],
    'batch_size': BATCH_SIZE,
    'n_window': N_WINDOW,
    'aggreg': 5,
    'input_size': INPUT_SIZE,
    'model_output_size': OUTPUT_SIZE,
    'target_output_size': TARGET_OUTPUT_SIZE,
    'sync_axis': 2,
    'stepy': STEPY,
    'edges': EDGES,
    'force_params': [K_C, M_C, K_N, M_N, K_T, M_T],
    'mother_wavelet': 'mexh',
    'fz_freq_divider': 30,
    'nb_scales': 84,
    'signif_lvl': 0.99,
    'random_seed': RANDOM_SEED
}

model_config = {
    'models_dir': MODELS_DIR,
    'activation': 'ReLU',
    'nb_units': 100,
    'nb_layers': 2,
    'kernel_size': 3,
    'padding': 1,
    'dropout_rate': 0.5,
    'batch_size': BATCH_SIZE,
    'input_size': INPUT_SIZE,
    'output_size': OUTPUT_SIZE,
    'target_output_size': TARGET_OUTPUT_SIZE,
    'init': 'kaiming_normal',
    'learning_rate': 0.0001,
    'max_iter': 50,
    'reg_lambda': 0.01,
    'loss': 'MSELoss',
    'force_samples': STEPY * EDGES,
    'random_seed': RANDOM_SEED
}
