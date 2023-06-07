"""Trainer class for force predictions"""

import numpy as np
from tqdm import tqdm
from pytorchutils.basic_trainer import BasicTrainer
from pytorchutils.globals import torch, DEVICE
from matplotlib import pyplot as plt
#plt.switch_backend('Agg')

class Trainer(BasicTrainer):
    """Wrapper class for force trainer"""
    def __init__(self, config, model, preprocessor):
        BasicTrainer.__init__(self, config, model, preprocessor)

    def learn_from_epoch(self, epoch_idx):
        """Training method"""
        epoch_loss = 0
        nb_scenarios = 0
        inp_scenario, out_scenario, rays_scenario = self.get_batches_fn()

        out_size = np.shape(out_scenario)[-1]
        plot_out = np.reshape(out_scenario, (-1, out_size))
        plot_pred = None

        while inp_scenario.any():
            force_samples = self.config['force_samples']
            batch_size = np.shape(inp_scenario)[1]
            summed_pred = np.empty((len(inp_scenario) * batch_size, out_size))

            for batch_idx in tqdm(range(len(inp_scenario))):
                pred_out = self.predict(inp_scenario[batch_idx])
                pred_out = self.postprocess(pred_out, rays_scenario, batch_idx)

                pred_sum = pred_out.detach().cpu().numpy()
                nb_pred_forces = batch_size
                summed_pred[
                    batch_idx * nb_pred_forces:batch_idx * nb_pred_forces + nb_pred_forces
                ] = pred_sum

                batch_loss = self.loss(
                    pred_out,
                    torch.Tensor(out_scenario[batch_idx]).to(DEVICE)
                )

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                epoch_loss += batch_loss.item()

            inp_scenario, out_scenario, rays_scenario = self.get_batches_fn()
            nb_scenarios += 1

            if nb_scenarios == 1:
                plot_pred = summed_pred

        epoch_loss /= nb_scenarios

        __, axs = plt.subplots(np.shape(plot_out)[-1], 1)
        for idx in range(np.shape(plot_out)[-1]):
            axs[idx].plot(
                plot_pred[len(plot_pred) // 2:len(plot_pred) // 2 + 1000, idx],
                label='Pred'
            )
            axs[idx].plot(
                plot_out[len(plot_out) // 2:len(plot_out) // 2 + 1000, idx],
                label='Ref'
            )
            axs[idx].legend()
        plt.savefig(
            '{}/epoch{}_loss={}_pred.png'.format(
                self.results_dir,
                self.current_epoch,
                epoch_loss
            )
        )
        return epoch_loss

    def predict(self, inp):
        """Capsuled prediction function"""
        inp = torch.Tensor(inp).to(DEVICE)
        # print(inp.size())
        # quit()
        pred_out = self.model(inp)
        return pred_out

    def postprocess(self, data, rays, batch_idx):
        """Preprocessing method"""
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

            f = torch.sum(
                torch.einsum('ik, ikl->il', pred, ray_info),
                dim=0
            )
            # print(f.size())


            forces[idx] = torch.sum(
                torch.einsum('ik, ikl->il', pred, ray_info),
                dim=0
            )
        return forces

    def evaluate(self, inp, out, rays, batch_idx):
        """Prediction and error estimation for given input and output"""
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
            forces = self.postprocess(pred_out, rays, batch_idx)

            # RMSE is the default accuracy metric
            error = torch.sqrt(
                self.loss(
                    forces,
                    torch.Tensor(out).to(DEVICE)
                )
            )
            return forces, (error * 100.0)
