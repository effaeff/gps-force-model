"""Config file"""

RANDOM_SEED = 1234

DATA_DIR = '/raid/gps-forces/6_features'
RESULTS_DIR = '/cephfs/gps-forces/results'
MODELS_DIR = '/cephfs/gps-forces/models/complex-shuffle-target0-drop'
PARAMS = f'{DATA_DIR}/LHS_Final_new.xlsx'

WINDOW = False
INPUT_SIZE = 3
OUTPUT_SIZE = 3
TARGET_OUTPUT_SIZE = 3
NB_MODELS = 1
STEPY = 75
EDGES = 2
BATCH_SIZE = 1024

####################################
######### Temporal shizzle #########
####################################
N_WINDOW = 7
KERNEL_SIZE = 3
DILATE = True
DILATION_BASE = 2
if DILATE:
    # Calculate the required number of layers for full coverage of window
    NB_LAYERS = int(np.ceil(
        (np.log(((N_WINDOW - 1) * (DILATION_BASE - 1)) / (KERNEL_SIZE - 1) + 1) / np.log(DILATION_BASE))
    ))
    # Exponentially increasing dilation
    # This also exponentially increases receptive field of stacked convolutions
    # and reduces the number of required layers for full coverage logarithmically
    DILATION = [DILATION_BASE**idx for idx in range(NB_LAYERS)]
else:
    # If no dilation is used, required number of layers increase to ensure full coverage of window
    # Therefore, nb_layers gets split between encoder and decoder
    NB_LAYERS = int(np.ceil((N_WINDOW - 1) / (KERNEL_SIZE - 1)))
    DILATION = [1 for __ in range(NB_LAYERS)]

# Begin with 32 channels and increase exponentially with base 2 for each layer
CHANNELS = [2**(5 + idx) for idx in range(NB_LAYERS)]
# Left padding for each layer considering dilation and kernel_size
PADDING = [d * (KERNEL_SIZE - 1) for d in DILATION]

# Config for non-window model
if not WINDOW:
    NB_LAYERS = 3
    DILATION = [1 for __ in range(NB_LAYERS)]
    PADDING = 1
    CHANNELS = [2**(5 + idx) for idx in range(NB_LAYERS)]

####################################

K_C = 272.94663445441347
K_N = 344.1285160407239
K_T = -50.57065214836188
M_C = 0.4118070182339672
M_N = 0.27309617279279746
M_T = 0.3288884071807737

data_config = {
    'random_seed': RANDOM_SEED,
    'keys': ['t', 'fx', 'fy', 'fz'],
    'group': 'Recorder',
    'data_dir': DATA_DIR,
    'results_dir': RESULTS_DIR,
    'params': PARAMS,
    'excel_sheet': 'LHS_new',
    # 'scaler': ['x_standard.scal', 'y_standard.scal'],
    # 'scaler': ['x_minmax.scal', 'y_minmax.scal'],
    'scaler': 'minmax.scal',
    'test_size': 0.2,
    'batch_size': BATCH_SIZE,
    'n_window': N_WINDOW,
    'input_size': INPUT_SIZE,
    'stepy': STEPY,
    'edges': EDGES,
    'aggreg': 5, # Not used
    # Parameters which are required if multiple models are desired
    'model_output_size': OUTPUT_SIZE,
    'target_output_size': TARGET_OUTPUT_SIZE,
    # The following parameters are for time series sync shit
    'negate': [-1, 1, 1],
    'sync_axis': 2,
    'force_params': [K_C, M_C, K_N, M_N, K_T, M_T],
    'mother_wavelet': 'mexh',
    'fz_freq_divider': 30,
    'nb_scales': 84,
    'signif_lvl': 0.99
}

model_config = {
    'random_seed': RANDOM_SEED,
    'models_dir': MODELS_DIR,
    'activation': 'ReLU', # Also tried PReLU and LeakyReLU
    # 'kernel_size': (KERNEL_SIZE, 1, 1),
    # 'kernel_size': (KERNEL_SIZE, 1),
    'kernel_size': KERNEL_SIZE,
    'padding': PADDING,
    'nb_layers': NB_LAYERS,
    'dilation': DILATION,
    'channels': CHANNELS,
    'batch_size': BATCH_SIZE,
    'input_size': INPUT_SIZE,
    'output_size': OUTPUT_SIZE,
    'n_window': N_WINDOW,
    'force_samples': STEPY * EDGES,
    'init': 'kaiming_normal',
    'learning_rate': 0.0001, # Also tried 0.00001, was worse
    'max_iter': 50,
    'reg_lambda': 0.001,
    'drop_rate': 0.2,
    # 'reg_lambda': 0,
    'loss': 'MSELoss',
    # Early stopper stuff
    'patience': 10,
    'max_problem': False,
    # Parameters which are required if multiple models are desired
    'nb_models': NB_MODELS,
    'target_output_size': TARGET_OUTPUT_SIZE,
}
