"""Script for evaluating already trained models"""

import re
import math
import os
import numpy as np
from sklearn.metrics import mean_squared_error

from config import RESULTS_DIR, TARGET_OUTPUT_SIZE

def main():
    """Main method"""
    approaches = os.listdir(RESULTS_DIR)

    for app_idx, approach in enumerate(approaches):
        epochs = sorted(
            os.listdir(f'{RESULTS_DIR}/{approach}'), key=lambda x: int(re.search('\d+', x).group())
        )
        errors = np.empty(len(epochs))
        stds = np.empty(len(epochs))
        for epoch_idx, epoch in enumerate(epochs):
            error = np.array([0, 0, 0])
            scenarios = [
                fname for fname in os.listdir(f'{RESULTS_DIR}/{approach}/{epoch}')
                if fname.endswith('.npy')
            ]
            epoch_errors = np.zeros((len(scenarios), TARGET_OUTPUT_SIZE))
            for scenario_idx, scenario in enumerate(scenarios):
                # Data is [target, pred] and has shape (2, N, 3)
                data = np.load(f'{RESULTS_DIR}/{approach}/{epoch}/{scenario}')
                for out_idx in range(TARGET_OUTPUT_SIZE):
                    epoch_errors[scenario_idx, out_idx] = math.sqrt(
                        mean_squared_error(data[0, :, out_idx], data[1, :, out_idx])
                    ) / np.ptp(data[0, :, out_idx]) * 100.0

            epoch_errors = np.mean(epoch_errors, axis=1)

            errors[epoch_idx] = epoch_errors.mean()
            stds[epoch_idx] = epoch_errors.std()

        min_idx = np.argmin(errors)
        print(
            f"Approach: {approach:<40}\t"
            f"NRMSE: {errors[min_idx]:.2f} +- {stds[min_idx]:.2f}\t"
            f"Best epoch: {epochs[min_idx]}"
        )

if __name__ == '__main__':
    main()
