"""Trainer class for force predictions"""

import numpy as np
from tqdm import tqdm
from pytorchutils.basic_trainer import BasicTrainer
from pytorchutils.globals import torch, DEVICE
from matplotlib import pyplot as plt
#plt.switch_backend('Agg')

from config import WINDOW

class Trainer(BasicTrainer):
    """Wrapper class for force trainer"""
    def __init__(self, config, model, preprocessor):
        BasicTrainer.__init__(self, config, model, preprocessor)
        self.output_size = config['output_size']
        self.nb_models = config.get('nb_models', 1)
        self.target_output_size = config['target_output_size']

    def learn_from_epoch(self, epoch_idx, verbose):
        """Training method"""
        epoch_loss = 0
        inp_scenario, out_scenario, rays_scenario, nb_scenarios = self.get_batches_fn()

        if verbose:
            pbar = tqdm(total=nb_scenarios, desc=f'Epoch {epoch_idx}', unit='scenario')

        scenario_idx = 0
        while inp_scenario.any():
            force_samples = self.config['force_samples']
            batch_size = np.shape(inp_scenario)[1]

            for batch_idx, inp_batch in enumerate(inp_scenario):
                pred_out = self.predict(inp_batch)

                # Windowing: reshape
                # from (batch_size, input_size, n_window, force_samples)
                # to   (batch_size * n_window, force_samples, input_size)
                # because effective batch size if batch_size * n_window
                if WINDOW:
                    pred_out = torch.moveaxis(
                        pred_out, 1, -1
                    ).reshape((-1, force_samples, self.output_size))

                pred_out = self.postprocess_batch(pred_out, rays_scenario, batch_idx)

                batch_loss = self.loss(
                    pred_out,
                    torch.from_numpy(out_scenario[batch_idx]).float().to(DEVICE)
                )

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                epoch_loss += batch_loss.item()


            if verbose:
                pbar.update(1)

            # self.validate(epoch_idx, True, verbose, f'{scenario_idx:03d}')

            inp_scenario, out_scenario, rays_scenario, __ = self.get_batches_fn()
            scenario_idx += 1

        if verbose:
            pbar.close()

        return epoch_loss

    def predict(self, inp):
        """Capsuled prediction function"""
        inp = torch.Tensor(inp).to(DEVICE)
        nb_models = self.config.get('nb_models', 1)
        force_samples = self.config['force_samples']
        if nb_models > 1:
            pred_out = torch.empty(
                len(inp),
                force_samples,
                nb_models # Multiple models are assumed to predict exactly one output for now
            ).to(DEVICE)
            for idx, __ in enumerate(self.model):
                local_pred = self.model[idx](inp)
                pred_out[:, :, idx] = local_pred
        else:
            pred_out = self.model(inp)
        return pred_out

    def postprocess(self, data, rays, batch_idx):
        """Preprocessing method using einsum for each force vector"""
        batch_size = len(data)
        force_samples = self.config['force_samples']
        pred_out_size = np.shape(data)[-1]
        forces = torch.empty(len(data), 3).to(DEVICE)

        for idx in range(0, len(data)):
            pred = data[idx]
            start_idx = (batch_idx * batch_size * force_samples + idx * force_samples) % len(rays)
            ray_info = (rays[start_idx:start_idx + force_samples] * (-1))
            ray_info = torch.tensor(
                np.reshape(
                    ray_info,
                    (force_samples, pred_out_size, pred_out_size)
                )
            ).to(DEVICE).float()

            forces[idx] = torch.sum(
                torch.einsum('ik, ikl->il', pred, ray_info),
                dim=0
            )
        return forces

    def postprocess_batch(self, pred_out, rays, batch_idx):
        """Postprocess whole batch of predictions at once using ray directions"""
        batch_size = len(pred_out)
        force_samples = self.config['force_samples']
        # Construct ray_info for complete batch by concatenating from the beginning
        # until all tool revolutions of batch are covered
        required_infos = batch_size * force_samples

        start_idx = (batch_idx * batch_size * force_samples) % len(rays)

        if (len(rays) - start_idx) >= required_infos:
            ray_info = rays[start_idx:start_idx + required_infos]
        else:
            ray_info = rays[start_idx:]

            covered = len(rays) - start_idx

            full_concats = (required_infos - covered) // len(rays)
            ray_info = np.concatenate(
                (
                    ray_info,
                    *[rays for __ in range(full_concats)],
                    rays[:(required_infos - (covered + full_concats * len(rays)))]
                )
            )
        out_dim = self.output_size * self.nb_models
        ray_info = np.reshape(ray_info, (batch_size, force_samples, out_dim, out_dim))

        pred_out = torch.sum(
            torch.einsum('bik, bikl->bil', pred_out, torch.from_numpy(ray_info).float().to(DEVICE)),
            dim=1
        )
        return pred_out

    def evaluate(self, inp, rays, batch_idx):
        """Prediction and error estimation for given input and output"""
        force_samples = self.config['force_samples']
        with torch.no_grad():
            # Switch to PyTorch's evaluation mode.
            # Some layers, which are used for regularization, e.g., dropout or batch norm layers,
            # behave differently, i.e., are turnd off, in evaluation mode
            # to prevent influencing the prediction accuracy.
            if isinstance(self.model, (list, np.ndarray)):
                for idx, __ in enumerate(self.model):
                    self.model[idx].eval()
            else:
                self.model.eval()
            pred_out = self.predict(inp)

            if WINDOW:
                pred_out = torch.moveaxis(
                    pred_out, 1, -1
                ).reshape((-1, force_samples, self.output_size))
            else:
                pred_out = pred_out.reshape((-1, force_samples, self.output_size))

            pred_out = self.postprocess_batch(pred_out, rays, batch_idx)

            return pred_out
