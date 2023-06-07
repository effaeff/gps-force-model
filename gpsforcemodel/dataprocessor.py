"""Collection of functions for data preprocessing"""

import re
import math
import glob
import sqlite3
from sqlite3 import Error
import os
import pywt
import pycwt
import tdms
import scipy.signal
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from openpyxl import load_workbook
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import joblib
# plt.switch_backend('Agg')


class DataProcessor:
    """Class to wrapp preprocessing methods and store dataset"""
    def __init__(self, config):
        self.config = config
        self.batch_size = self.config['batch_size']
        self.n_window = self.config.get('n_window', 32)
        self.stepy = self.config['stepy']
        self.edges = self.config['edges']
        self.input_size = self.config['input_size']
        self.model_output_size = self.config['model_output_size']
        self.target_output_size = self.config['target_output_size']
        self.prediction_output_size = self.model_output_size
        self.random_seed = self.config['random_seed']
        self.sql_command = (
            'SELECT phi, kappa, uncutThickness, '
            'dc_x, dc_y, dc_z, dn_x, dn_y, dn_z, dt_x, dt_y, dt_z, '
            'uncutWidth, cuttingSpeed, helixAngle, relHeight '
            'FROM ForceModelParameters'
        )
        self.force_params = self.config['force_params']
        self.keys = self.config['keys']
        self.group = self.config['group']
        self.sync_axis = self.config['sync_axis']
        self.force_samples = self.edges * self.stepy
        self.data_dir = self.config['data_dir']

        # Get pathnames of files
        files = sorted(
            [
                fname for fname in os.listdir(self.data_dir)
                if fname.endswith('features_aggreg.npy')
            ], key=lambda x: int(re.search('\d+', x.split('_')[4]).group())
        )
        # files = glob.glob(os.path.join(self.config['data_dir'], '*.db'))

        self.parameters_file = self.config['params']
        excel_sheet_label = self.config['excel_sheet']

        np.random.seed(self.random_seed)

        excel_sheet = load_workbook(filename=self.parameters_file)[excel_sheet_label]
        skiprows = 1
        columns = 7
        self.parameter_values = dict()
        for idx, row in enumerate(excel_sheet.values):
            if idx >= skiprows:
                data = row[:columns]
                spsp = data[0]
                f_z = data[2]
                alpha = data[3]
                exp_number = data[4]
                substeps = data[6]
                self.parameter_values[exp_number] = [spsp, f_z, alpha, substeps]
        # for idx in tqdm(range(len(files))):
            # self.process_file(files[idx])

        self.train_files, self.test_files = train_test_split(
            files,
            test_size=self.config['test_size'],
            random_state=self.config['random_seed']
        )
        self.val_files, self.test_files = train_test_split(
            self.test_files,
            test_size=0.5,
            random_state=self.config['random_seed']
        )

        # train_number_substeps = [
        #     (
        #         self.parameter_values[
        #             int(
        #                 re.search(
        #                     r'\d+', os.path.basename(filename).split('_')[-1]
        #                 ).group()
        #             )
        #         ][3],
        #         filename
        #     )
        #     for filename in self.train_files
        # ]

        # train_number_substeps.sort(key=lambda elem: elem[0])
        # self.train_files = [elem[1] for elem in train_number_substeps]

        self.train_numbers = [
            int(
                re.search(
                    r'\d+', os.path.basename(filename).split('_')[4]
                ).group()
            )
            for filename in self.train_files
        ]

        self.val_numbers = [
            int(
                re.search(
                    r'\d+', os.path.basename(filename).split('_')[4]
                ).group()
            )
            for filename in self.val_files
        ]

        # Init scaler
        scaler_filenames = self.config.get('scaler', '')
        scaler_exists = True
        if scaler_filenames:
            for filename in scaler_filenames:
                scaler_exists &= os.path.isfile(filename)
        else:
            scaler_exists = False
        self.x_scaler = StandardScaler(copy=False)
        self.y_scaler = StandardScaler(copy=False)
        if scaler_exists:
            self.x_scaler = joblib.load(scaler_filenames[0])
            self.y_scaler = joblib.load(scaler_filenames[1])
            self.scaler_loaded = True
        else:
            print("Loading full dataset to fit scaler")
            print("Fitting scaler...")
            for idx in tqdm(range(len(self.train_files))):
                x__ = np.load(
                    f'{self.data_dir}/'
                    f'finkeldey_sfb876_kf10_ae35_nr{self.train_numbers[idx]:03d}_features_aggreg.npy'
                )
                y__ = np.load(
                    f'{self.data_dir}/'
                    f'finkeldey_sfb876_kf10_ae35_nr{self.train_numbers[idx]:03d}_target_aggreg.npy'
                )
                self.x_scaler.partial_fit(x__)
                self.y_scaler.partial_fit(y__)
            joblib.dump(self.x_scaler, 'x_scaler.scal')
            joblib.dump(self.y_scaler, 'y_scaler.scal')

        self.scenario_idx = 0

    def analyze_data(self):
        """Analyze distribution of data"""
        features = []
        for idx in tqdm(range(len(self.train_files))):
            x__ = np.load('{}_features.npy'.format(self.train_files[idx]))
            features.append(x__)
        features = np.vstack(features) # (h, b, vc, spsp, fz, alpha)
        nb_bins = 20
        nb_feat = np.shape(features)[-1]
        labels = [
            'Chip thickness',
            'Chip width',
            'Cutting speed',
            'Spindle speed',
            'Feed per tooth',
            'Inclination angle'
        ]
        __, axs = plt.subplots(1, nb_feat, figsize=(10, 10))
        for idx in range(nb_feat):
            N, __, patches = axs[idx].hist(features[:, idx], bins=nb_bins)
            # We'll color code by height, but you could use any scalar
            fracs = N / N.max()

            # we need to normalize the data to 0..1 for the full range of the colormap
            norm = colors.Normalize(fracs.min(), fracs.max())

            # Now, we'll loop through our objects and set the color of each accordingly
            for thisfrac, thispatch in zip(fracs, patches):
                color = plt.cm.viridis(norm(thisfrac))
                thispatch.set_facecolor(color)
            axs[idx].set_xlabel(labels[idx])
            axs[idx].set_ylabel('Distribution')
            axs[idx].yaxis.set_major_formatter(PercentFormatter(xmax=N.max()))
        plt.tight_layout()
        plt.savefig('distribution.png', dpi=600)

    def get_next_batches(self):
        """Returns next batched scenario based on scenario index"""
        # All scenarios where processed. Next epoch has to be initiated
        if self.scenario_idx >= len(self.train_files):
            self.scenario_idx = 0

            # Randomize order of scenarios for next epoch
            train = list(zip(self.train_files, self.train_numbers))
            np.random.shuffle(train)
            self.train_files, self.train_numbers = zip(*train)

            # Signal to trainer that current epoch is finished
            return np.empty(0), np.empty(0), np.empty(0)

        # Select next scenario of current epoch based on self.scenario_idx
        print("Scenario {}/{}: {}".format(
            self.scenario_idx + 1,
            len(self.train_files),
            self.train_files[self.scenario_idx]
        ))

        # Bachify current scenario
        x__, y__, rays = self.prepare_batches(
            self.train_files[self.scenario_idx],
            self.train_numbers[self.scenario_idx]
        )
        self.scenario_idx += 1

        return x__, y__, rays

    def prepare_batches(self, filename, number):
        """Load, aggregate, truncate, scale and reshape preprocessed and saved data"""
        # stepx = 2 * self.parameter_values[number][-1]
        # batch_size = int(self.config['batched_turnarounds'] * stepx * self.feed_samples)
        batch_size = self.batch_size

        x__ = np.load(
            f'{self.data_dir}/'
            f'finkeldey_sfb876_kf10_ae35_nr{number:03d}_features_aggreg.npy'.format(filename)
        )
        rays = np.load(
            f'{self.data_dir}/'
            f'finkeldey_sfb876_kf10_ae35_nr{number:03d}_rays_aggreg.npy'.format(filename)
        )
        y__ = np.load(
            f'{self.data_dir}/'
            f'finkeldey_sfb876_kf10_ae35_nr{number:03d}_target_aggreg.npy'.format(filename)
        )

        # print(np.shape(x__))
        # print(np.shape(y__))

        # out_batch_size = batch_size #// self.force_samples

        # x__, y__ = self.aggregate(x__, y__)
        # x__ = np.reshape(
            # self.x_scaler.transform(
                # x__[:(len(x__) // (self.n_window * batch_size)) * self.n_window * batch_size]
            # ),
            # (-1, batch_size, self.n_window, np.shape(x__)[-1])
        # )
        # y__ = np.reshape(
            # self.y_scaler.transform(
                # y__[
                    # :(
                        # (len(y__) // ((self.n_window//self.force_samples) * batch_size)) *
                        # (self.n_window//self.force_samples) * batch_size
                    # )
                # ]
            # ),
            # (-1, out_batch_size, self.n_window // self.force_samples, np.shape(y__)[-1])
        # )

        ################
        ### New idea ###
        ################

        # Store features of all force samples in one image-like structure
        # As a consequence, the shape of each batch should be (B, force_samples, input_size)

        substeps = self.parameter_values[number][-1] // 5

        # Scale and reshape into images of size (force_samples, input_size)
        x__ = np.reshape(
            self.x_scaler.transform(
                x__[:(len(x__) // (batch_size * self.force_samples) * (batch_size * self.force_samples))]
            ),
            (-1, self.batch_size, 1, self.force_samples, self.input_size)
        )

        y__ = np.reshape(
            self.y_scaler.transform(
                y__[:(len(y__) // batch_size * batch_size)]
            ),
            (-1, self.batch_size, y__.shape[-1])
        )

        ################

        return x__, y__, rays


    def aggreg_all(self):
        """Aggregate full data set"""
        full_data = np.concatenate((self.train_files, self.val_files))

        for filename in full_data:
            print("Aggregate file: {}".format(filename))
            x__ = np.load('{}_features.npy'.format(filename))
            y__ = np.load('{}_target.npy'.format(filename))
            rays = np.load('{}_rays.npy'.format(filename))
            x_aggreg, y_aggreg, rays_aggreg = self.aggregate(x__, y__, rays)
            np.save('{}_features_aggreg.npy'.format(os.path.splitext(filename)[0]), x_aggreg)
            np.save('{}_target_aggreg.npy'.format(os.path.splitext(filename)[0]), y_aggreg)
            np.save('{}_rays_aggreg.npy'.format(os.path.splitext(filename)[0]), rays_aggreg)

    def aggregate(self, x__, y__, rays):
        """Aggregate data"""
        aggreg = self.config['aggreg']
        x_aggreg = np.empty((len(x__) // aggreg, np.shape(x__)[-1]))
        x_cutoff = len(x__) // (self.force_samples * aggreg) * (self.force_samples * aggreg)
        y_cutoff = len(y__) // aggreg * aggreg
        rays_cutoff = len(rays) // (self.force_samples * aggreg) * (self.force_samples * aggreg)
        x__ = x__[:x_cutoff]
        y__ = y__[:y_cutoff]
        rays = rays[:rays_cutoff]

        # Features
        for jdx in tqdm(range(0, len(x__), self.force_samples * aggreg)):
            x_slice = x__[jdx:jdx + self.force_samples * aggreg]
            for kdx in range(self.force_samples):
                for ldx in range(np.shape(x__)[-1]):
                    x_mean = np.mean(
                        x_slice[kdx:kdx + self.force_samples * aggreg:self.force_samples, ldx]
                    )
                    x_aggreg[jdx // aggreg + kdx, ldx] = x_mean
        # # Target
        y_aggreg = np.empty((len(y__) // aggreg, np.shape(y__)[-1]))
        for jdx in range(0, len(y__), aggreg):
            for ldx in range(np.shape(y__)[-1]):
                y_aggreg[jdx // aggreg, ldx] = np.mean(y__[jdx:jdx + aggreg, ldx])
        # Rays
        rays_aggreg = np.empty((len(rays) // aggreg, np.shape(rays)[-1]))
        for jdx in tqdm(range(0, len(rays), self.force_samples * aggreg)):
            rays_slice = rays[jdx:jdx + self.force_samples * aggreg]
            for kdx in range(self.force_samples):
                for ldx in range(np.shape(rays)[-1]):
                    rays_mean = np.mean(
                        rays_slice[kdx:kdx + self.force_samples * aggreg:self.force_samples, ldx]
                    )
                    rays_aggreg[jdx // aggreg + kdx, ldx] = rays_mean

        return x_aggreg, y_aggreg, rays_aggreg

    def read_db(self, filename, command):
        """Fetch data from db file using given sql command"""
        try:
            conn = sqlite3.connect(filename)
            with conn:
                cur = conn.cursor()
                cur.execute(command)
                rows = cur.fetchall()
                return rows
        except Error:
            print(Error)

        return None

    def find_peak(self, array):
        """Find falling edge in rect signals"""
        peak_slice = np.empty(len(array))
        cut = False
        peak_idx = 0
        for jdx, value in enumerate(array):
            if value == 1 and not cut:
                cut = True
                peak_slice[jdx] = 0
            elif value == 0 and cut:
                peak_slice[jdx] = 1
                peak_idx = jdx
                peak_slice[jdx + 1:len(array)] = [0 for __ in range(len(array) - jdx - 1)]
                break
            else:
                peak_slice[jdx] = 0
        return peak_slice, peak_idx

    def find_mid_peak(self, array):
        """Find index, which corresponds to the middle of a rect signal"""
        cut = False
        start_idx = 0
        for idx, value in enumerate(array):
            if value == 1 and not cut:
                cut = True
                start_idx = idx
            elif value == 0 and cut:
                return start_idx + (idx - start_idx) // 2

    def process_file(self, filename):
        """Method for processing a single file"""
        number = os.path.splitext(os.path.basename(filename))[0].split('_')[-1]
        print(number)
        number = int(re.search(r'\d+', number).group())
        data = self.read_db(filename, self.sql_command)
        # cutoff = (len(data) // self.batch_size) * self.batch_size
        # data = data[:cutoff]
        # Number of steps per tooth feed
        substeps = data[self.stepy][0]
        # Batchified features
        batches = np.empty((
            len(data),
            self.input_size
        ))
        # Batchified ray infos
        total_ray_d = np.empty((
            len(data),
            self.target_output_size * self.prediction_output_size
        ))
        # Batchified simulated forces
        targets = np.empty((
            len(data) // self.force_samples,
            self.target_output_size
        ))
        # Since it cannot be ensured
        # that the batch size is a multiple of substeps * stepy * #edges,
        # several indices have to be used to fill the resulting arrays.
        batches_idx = 0
        targets_idx = 0
        ray_infos_idx = 0
        #batch = np.empty((self.batch_size, self.input_size))
        #ray_d = np.empty(
            #(self.batch_size, self.target_output_size * self.prediction_output_size)
        #)
        #ray_info_idx = 0
        #batch_idx = 0
        print("Batchify file: {}...".format(filename))

        parameters = self.parameter_values[number]
        spsp = parameters[0]
        f_z = parameters[1]
        alpha = parameters[2]
        print("Parameter values: Spindle speed: {}, Tooth feed: {}, Inclination angle: {}".format(
            spsp,
            f_z,
            alpha
        ))

        for revolution in tqdm(range(0, len(data), self.force_samples)):
            # Variable for summing the simulated force vector for each time step
            force = np.zeros(self.target_output_size)
            # There are multiple feature vectors for each force vector
            for ray in range(self.force_samples):
                index = revolution + ray
                # Features
                chip_thickness = data[index][2]
                chip_width = data[index][12] if chip_thickness > 0 else 0.0
                v_c = data[index][13] if chip_thickness > 0 else 0.0
                batches[batches_idx] = np.array([chip_thickness, v_c, spsp, f_z])
                batches_idx += 1
                # Ray directions
                d_c = np.array([data[index][3], data[index][4], data[index][5]])
                d_n = np.array([data[index][6], data[index][7], data[index][8]])
                d_t = np.array([data[index][9], data[index][10], data[index][11]])
                ray_info = np.array([
                    d_c[0],
                    d_c[1],
                    d_c[2],
                    d_n[0],
                    d_n[1],
                    d_n[2],
                    d_t[0],
                    d_t[1],
                    d_t[2]
                ])
                total_ray_d[ray_infos_idx] = ray_info
                ray_infos_idx += 1
                # Simulated forces
                f_c = (-1) * d_c * self.force_params[0] * data[index][12] * math.pow(
                    data[index][2], (1 - self.force_params[1])
                )
                f_n = (-1) * d_n * self.force_params[2] * data[index][12] * math.pow(
                    data[index][2], (1 - self.force_params[3])
                )
                f_t = (-1) * d_t * self.force_params[4] * data[index][12] * math.pow(
                    data[index][2], (1 - self.force_params[5])
                )
                force += (f_c + f_n + f_t)
            targets[targets_idx] = force
            targets_idx += 1
            # Check if the current batch is full
            # if batch_idx >= self.batch_size:
            #     batches[batches_idx] = batch
            #     batches_idx += 1
            #     batch = np.empty((self.batch_size, self.input_size))
            #     batch_idx = 0
            #     total_ray_d[ray_infos_idx] = ray_d
            #     ray_infos_idx += 1
            #     ray_d = np.empty((self.batch_size, 9))
            #     ray_info_idx = 0

        # Read corresponding measured forces
        exp_filename = '{}.tdms'.format(os.path.splitext(filename)[0])
        print("Reading experimental file: {}".format(exp_filename))
        forces_exp = tdms.read_tdms(
            exp_filename,
            self.keys,
            self.group
        )
        # Resample measured forces to the sample frequency of the simulated data
        print("Resampling measurements...")
        revolutions_per_second = spsp / 60.0
        tooth_engagements_per_second = revolutions_per_second * self.edges
        sample_freq = substeps * tooth_engagements_per_second
        time_diff_exp = forces_exp[0][-1] - forces_exp[0][0]
        # Cut off time channel
        forces_exp = forces_exp[1:]
        for idx, __ in enumerate(forces_exp):
            forces_exp[idx] = scipy.signal.resample(
                forces_exp[idx],
                int(time_diff_exp * sample_freq)
            )
        # Sample synchronization between simulated data and measurements

        # used_samples = int(self.config['used_data_portion'] * len(targets))
        # sim_cutoff = (len(targets) - used_samples) * feed_samples
        # sim_cutoff = (sim_cutoff // self.batch_size) * self.batch_size
        # targets = targets[sim_cutoff // feed_samples:]
        # batches = batches[sim_cutoff:]
        # total_ray_d = total_ray_d[sim_cutoff:]
        # exp_cutoff = len(forces_exp[self.sync_axis]) - used_samples
        # for idx, __ in enumerate(forces_exp):
        #     forces_exp[idx] = forces_exp[idx][exp_cutoff:]

        sync_forces_sim = targets[:, self.sync_axis]


        fz_freq_div = self.config['fz_freq_divider']
        nb_scales = self.config['nb_scales']
        wavelet = self.config['mother_wavelet']
        signif_lvl = self.config.get('signif_lvl', 0.99)
        mother = None
        if wavelet == 'mexh':
            mother = pycwt.DOG(m=2)

        d_t = 1 / sample_freq
        fz_period = 1 / (spsp / 60.0 * self.edges)
        fz_freq = 1 / fz_period
        time_exp = np.array(
            [1 / sample_freq * idx for idx, __ in enumerate(forces_exp[self.sync_axis])]
        )
        time_sim = np.array([1 / sample_freq * idx for idx, __ in enumerate(sync_forces_sim)])

        segment_len = int(fz_period * sample_freq * 2)
        for channel_idx, channel in enumerate(forces_exp):
            predictors = np.empty(len(channel))
            for idx in range(0, len(channel), segment_len):
                segment = channel[idx:idx + segment_len]
                quartile = np.quantile(segment, 0.5)
                for jdx, __ in enumerate(segment):
                    index = idx + jdx
                    if segment[jdx] < quartile:
                        predictors[index] = segment[jdx]
                    else:
                        predictors[index] = quartile
            poly_predictors_coeffs = np.polyfit(time_exp - time_exp[0], predictors, 4)
            poly_predictors = np.polyval(poly_predictors_coeffs, time_exp - time_exp[0])
            exp_notrend = channel - poly_predictors
            std_exp = np.std(exp_notrend)
            exp_norm = exp_notrend / std_exp
            forces_exp[channel_idx] = exp_norm
        sync_forces_exp = forces_exp[self.sync_axis]

        freqs = [fz_freq / fz_freq_div + idx * fz_freq / fz_freq_div for idx in range(nb_scales)]
        central_freq = pywt.central_frequency(wavelet)
        scale = [central_freq / (d_t * value) for value in freqs]

        wave, scales, freqs, __, __, __ = pycwt.cwt(
            sync_forces_exp,
            d_t,
            s0=-1,
            wavelet=mother,
            freqs=np.array(freqs)
        )
        power = (np.abs(wave)) ** 2
        signif, __ = pycwt.significance(
            1.0,
            d_t,
            scales,
            0,
            0,
            significance_level=signif_lvl,
            wavelet=mother
        )
        sig99 = np.ones([1, len(sync_forces_exp)]) * signif[:, None]
        sig99 = power / sig99

        period = 1 / freqs
        # fig, axs = plt.subplots(2, 1, sharex=True)
        # cmap = axs[0].contourf(time_exp, period, wave, extend='both', cmap='inferno')
        # extent = [time_exp.min(), time_exp.max(), 0, max(period)]
        # axs[0].contour(time_exp, period, sig99, [-99, 1], colors='k', linewidths=2, extent=extent)
        # axs[0].plot(time_exp, [1 / fz_freq for __ in time_exp])
        # axs[1].plot(time_exp, sig99[fz_freq_div - 1], label='fz freq')
        # axs[1].plot(time_exp, [1 for __ in time_exp])
        signif_max_idx = np.argmax(sig99[fz_freq_div - 1])
        # jdx = 0
        # for idx in range(signif_max_idx, len(sig99[fz_freq_div - 1]), substeps):
        #     color = 'yellow'
        #     if jdx % 2 == 0:
        #         color = 'red'
        #     axs[0].plot((time_exp[idx], time_exp[idx]), (0, 0.01), color=color)
        #     jdx += 1
        # jdx = 0
        # for idx in range(signif_max_idx, 0, -substeps):
        #     color = 'yellow'
        #     if jdx % 2 == 0:
        #         color = 'red'
        #     axs[0].plot((time_exp[idx], time_exp[idx]), (0, 0.01), color=color)
        #     jdx += 1
        # jdx = 0
        # for idx in range(signif_max_idx, len(sig99[fz_freq_div - 1]), substeps):
        #     color = 'yellow'
        #     if jdx % 2 == 0:
        #         color = 'red'
        #     axs[1].plot((time_exp[idx], time_exp[idx]), (0, 5), color=color)
        #     jdx += 1
        # jdx = 0
        # for idx in range(signif_max_idx, 0, -substeps):
        #     color = 'yellow'
        #     if jdx % 2 == 0:
        #         color = 'red'
        #     axs[1].plot((time_exp[idx], time_exp[idx]), (0, 5), color=color)
        #     jdx += 1
        # axs[1].plot(time_exp, sync_forces_exp, label='exp')
        # axs[0].plot((time_exp[signif_max_idx], time_exp[signif_max_idx]), (0, 0.02))

        first_peak_idx = np.argmax(sig99[fz_freq_div - 1, :2 * substeps])
        second_peak_idx = first_peak_idx + substeps
        if first_peak_idx >= substeps:
            second_peak_idx = first_peak_idx - substeps

        # axs[0].plot((time_exp[first_peak_idx], time_exp[first_peak_idx]), (0, 0.03))
        # axs[0].plot((time_exp[second_peak_idx], time_exp[second_peak_idx]), (0, 0.02))


        crit_idx = 0
        for idx in range(signif_max_idx, 0, -(2 * substeps)):
            if sig99[fz_freq_div - 1, idx] >= 1 and wave[fz_freq_div - 1, idx + substeps] > 0:
                continue
            else:
                crit_idx = idx + 2 * substeps
                break

        # axs[1].plot((time_exp[crit_idx], time_exp[crit_idx]), (0, 10), color='blue')
        # plt.legend()
        # plt.colorbar(cmap, orientation='horizontal', pad=0.2)
        # plt.tight_layout()
        # plt.show()

        sync_forces_exp = sync_forces_exp[crit_idx:]
        sync_forces_sim = sync_forces_sim[crit_idx:]
        batches = batches[crit_idx * self.force_samples:]
        total_ray_d = total_ray_d[crit_idx * self.force_samples:]
        for idx, __ in enumerate(forces_exp):
            forces_exp[idx] = forces_exp[idx][crit_idx:]
        time_exp = time_exp[crit_idx:]
        time_sim = time_sim[crit_idx:]

        cwtmatr_exp, __ = pywt.cwt(sync_forces_exp, scale, wavelet, sampling_period=d_t)
        cwtmatr_sim, __ = pywt.cwt(sync_forces_sim, scale, wavelet, sampling_period=d_t)

        cwt_fz_exp = [1 if value > 0 else 0 for value in cwtmatr_exp[fz_freq_div - 1]]
        cwt_fz_sim = [1 if value > 0 else 0 for value in cwtmatr_sim[fz_freq_div - 1]]

        # fig, axs = plt.subplots(2, 2, sharex=True)
        # axs[0][0].plot(time_exp, sync_forces_exp)
        # axs[0][0].plot(time_exp, cwt_fz_exp)
        # axs[0][1].plot(time_sim, sync_forces_sim)
        # axs[0][1].plot(time_sim, cwt_fz_sim)
        # cmap = axs[1][0].contourf(time_exp, freqs, cwtmatr_exp, extend='both', cmap='inferno')
        # cmap_sim = axs[1][1].contourf(time_sim, freqs, cwtmatr_sim, extend='both', cmap='inferno')
        # axs[1][0].plot(time_exp, [fz_freq for __ in time_exp])
        # axs[1][1].plot(time_sim, [fz_freq for __ in time_sim])
        # fig.colorbar(cmap, orientation='horizontal', ax=axs[1][0])
        # fig.colorbar(cmap_sim, orientation='horizontal', ax=axs[1][1])
        # plt.tight_layout()
        # plt.show()

        sync_forces_exp = cwt_fz_exp
        sync_forces_sim = cwt_fz_sim

        # plt.plot(sync_forces_exp)
        # plt.plot(sync_forces_sim)
        # plt.show()

        # Determine the shorter time series
        end = (
            len(sync_forces_sim)
            if len(sync_forces_sim) < len(sync_forces_exp)
            else len(sync_forces_exp)
        )

        end = (end // substeps) * substeps
        offsets = []
        for idx in range(0, end, substeps):
            slice_sim = np.array(sync_forces_sim[idx:idx + substeps])
            slice_exp = np.array(sync_forces_exp[idx:idx + substeps])
            __, index_sim = self.find_peak(slice_sim)
            __, index_exp = self.find_peak(slice_exp)
            offset = index_exp - index_sim
            offsets.append(offset)

        cutoff = int(np.mean(offsets))
        if cutoff < 0:
            cutoff *= -1
            sync_forces_sim = sync_forces_sim[cutoff:]
            batches = batches[cutoff * self.force_samples:]
            total_ray_d = total_ray_d[cutoff * self.force_samples:]
        elif cutoff > 0:
            sync_forces_exp = sync_forces_exp[cutoff:]
            for idx, __ in enumerate(forces_exp):
                forces_exp[idx] = forces_exp[idx][cutoff:]

        start = False
        start_idx = 0

        for idx in range(2 * substeps):
            if sync_forces_exp[idx] == 1 and sync_forces_sim[idx] == 1:
                sync_forces_exp = sync_forces_exp[idx:]
                sync_forces_sim = sync_forces_sim[idx:]
                for jdx, __ in enumerate(forces_exp):
                    forces_exp[jdx] = forces_exp[jdx][idx:]
                batches = batches[idx * self.force_samples:]
                total_ray_d = total_ray_d[idx * self.force_samples:]
                break

        for idx in range(2 * substeps):
            if sync_forces_exp[idx] == 0 and sync_forces_sim[idx] == 0 and not start:
                start = True
                start_idx = idx
            elif (sync_forces_exp[idx] != 0 or sync_forces_sim[idx] != 0) and start:
                diff = idx - start_idx

                # plt.plot(exp_smoothed[:2 * substeps], label='exp')
                # plt.plot(sim_scaled[:2 * substeps], label='sim')
                # plt.plot((start_idx, start_idx), (0, 2))
                # plt.plot((idx, idx), (0, 2))
                # plt.legend()
                # plt.show()
                sync_forces_exp = sync_forces_exp[start_idx + diff // 2:]
                sync_forces_sim = sync_forces_sim[start_idx + diff // 2:]
                for jdx, __ in enumerate(forces_exp):
                    forces_exp[jdx] = forces_exp[jdx][start_idx + diff // 2:]
                batches = batches[(start_idx + diff // 2) * self.force_samples:]
                total_ray_d = total_ray_d[(start_idx + diff // 2) * self.force_samples:]
                break

        # Determine the shorter time series
        end = (
            len(sync_forces_sim)
            if len(sync_forces_sim) < len(sync_forces_exp)
            else len(sync_forces_exp)
        )
        end = (end // substeps) * substeps

        # plt.plot(sync_forces_sim, label='sim')
        # plt.plot(sync_forces_exp, label='exp')
        # plt.legend()
        # plt.show()

        # Negate experimental force directions so that the different coordinate systems match
        for idx, __ in enumerate(forces_exp):
            forces_exp[idx] *= self.config['negate'][idx]
        forces_exp = np.transpose(forces_exp)

        sim_synced = np.empty(end)
        exp_synced = np.empty(end)
        batches_synced = np.empty((end * self.force_samples, self.input_size))
        rays_synced = np.empty(
            (
                end * self.force_samples,
                self.target_output_size * self.prediction_output_size
            )
        )
        exp_forces_synced = np.empty((end, self.target_output_size))

        print("Synchronizing time series...")
        for idx in tqdm(range(0, end, substeps)):
            slice_sim = np.array(sync_forces_sim[idx:idx + substeps])
            slice_exp = np.array(sync_forces_exp[idx:idx + substeps])
            # plt.plot(slice_sim, label='sim_sync')
            # plt.plot(slice_exp, label='exp_sync')
            # local_forces = self.calc_forces_from_batches(
            #     batches[idx * feed_samples:(idx + substeps) * feed_samples],
            #     feed_samples,
            #     total_ray_d[idx * feed_samples:(idx + substeps) * feed_samples]
            # )
            # plt.plot(forces_exp[idx:idx + substeps, self.sync_axis], label='exp')
            # plt.plot(local_forces[:, self.sync_axis], label='sim')
            # plt.legend()
            # plt.figure()

            index_sim = self.find_mid_peak(slice_sim)
            index_exp = self.find_mid_peak(slice_exp)
            offset = index_exp - index_sim
            # print(offset)
            if offset < 0:
                slice_sim = np.roll(slice_sim, offset)
                sim_synced[idx:idx + substeps] = slice_sim
                exp_synced[idx:idx + substeps] = slice_exp
                batches_synced[
                    idx * self.force_samples:(idx + substeps) * self.force_samples
                ] = np.roll(
                    batches[idx * self.force_samples:(idx + substeps) * self.force_samples],
                    offset * self.force_samples,
                    axis=0
                )
                rays_synced[
                    idx * self.force_samples:(idx + substeps) * self.force_samples
                ] = np.roll(
                    total_ray_d[idx * self.force_samples:(idx + substeps) * self.force_samples],
                    offset * self.force_samples,
                    axis=0
                )
                exp_forces_synced[idx:idx + substeps] = forces_exp[idx:idx + substeps]
            else:
                slice_exp = np.roll(slice_exp, -offset)
                sim_synced[idx:idx + substeps] = slice_sim
                exp_synced[idx:idx + substeps] = slice_exp
                batches_synced[
                    idx * self.force_samples:(idx + substeps) * self.force_samples
                ] = batches[
                    idx * self.force_samples:(idx + substeps) * self.force_samples
                ]
                rays_synced[
                    idx * self.force_samples:(idx + substeps) * self.force_samples
                ] = total_ray_d[
                    idx * self.force_samples:(idx + substeps) * self.force_samples
                ]
                exp_forces_synced[idx:idx + substeps] = np.roll(
                    forces_exp[idx:idx + substeps],
                    -offset,
                    axis=0
                )
            # plt.plot(slice_sim, label='sim_sync')
            # plt.plot(slice_exp, label='exp_sync')
            # local_forces_synced = self.calc_forces_from_batches(
            #     batches_synced[idx * feed_samples:(idx + substeps) * feed_samples],
            #     feed_samples,
            #     rays_synced[idx * feed_samples:(idx + substeps) * feed_samples]
            # )
            # print(offset)
            # plt.plot(exp_forces_synced[idx:idx + substeps, self.sync_axis], label='exp')
            # plt.plot(local_forces_synced[:, self.sync_axis], label='sim')
            # plt.legend()
            # plt.show()

        # plt.plot(sim_synced, label='sim')
        # plt.plot(exp_synced, label='exp')
        # plt.show()

        # sync_samples = int(fz_period * sample_freq * self.sync_feeds)
        # print(end)
        # print(fz_period)
        # print(sync_samples)
        # print(substeps)
        # print(fz_period * sample_freq)

        # for idx in tqdm(range(0, end, sync_samples)):
        #     sync_batch_sim = sync_forces_sim[idx:idx + sync_samples]
        #     sync_batch_exp = sync_forces_exp[idx:idx + sync_samples]
        #     # Use the dynamic time warp algorithm
        #     # to generate the path of corresponding samples
        #     __, opt_path = series.dtw(
        #         sync_batch_sim,
        #         sync_batch_exp,
        #         lambda x, y: np.abs(x - y),
        #         win=[0, substeps]
        #     )
        #     # Use the maximum number of shifted samples along the path as offset
        #     cnt = [len(list(group)) for __, group in groupby(opt_path[0])]
        #     offsets.append(np.max(abs(np.array(opt_path[0]) - np.array(opt_path[1]))))
        #     #offsets.append(np.max(cnt))


        # Use mean of offsets
        # offset = int(np.mean(offsets))
        # print(offset)

        # plt.plot(sync_forces_sim)
        # plt.plot(sync_forces_exp[offset:])
        # plt.show()

        # Shift measurements with offset to align the time series
        # plot_forces = forces_exp[:, self.sync_axis]
        # Trim the time series to the same length
        # len_sim_forces = len(batches) // feed_samples
        # sample_diff = abs(len(forces_exp) - len_sim_forces)
        # exp_cutoff = len_sim_forces
        # print(len(batches))
        # if len(forces_exp) < len_sim_forces:
        #     batch_cutoff = (sample_diff * feed_samples) // (self.batch_size) + 1
        #     batches = batches[:-(batch_cutoff * self.batch_size)]
        #     total_ray_d = total_ray_d[:-(batch_cutoff * self.batch_size)]
        #     exp_cutoff = len(batches) // feed_samples
        # print(len(batches))
        # forces_exp = forces_exp[:exp_cutoff]

        # Batchify measurements
        # exp_batchified = np.reshape(
        #     forces_exp,
        #     (
        #         len(forces_exp) // (self.batch_size // feed_samples),
        #         self.batch_size // feed_samples,
        #         self.target_output_size
        #     )
        # )
        exp_batchified = exp_forces_synced
        batches = batches_synced
        total_ray_d = rays_synced

        forces = self.calc_forces_from_batches(batches, self.force_samples, total_ray_d)

        plt.title(number)
        plt.plot(
            np.reshape(
                exp_batchified,
                (-1, self.target_output_size)
            )[:, self.sync_axis],
            label='exp forces'
        )
        plt.plot(exp_synced, label='exp sync')
        plt.plot(np.array(forces)[:, self.sync_axis], label='sim forces')
        plt.plot(sim_synced, label='sim sync')
        plt.legend()
        plt.savefig('number_{}.png'.format(number), dpi=600)
        # plt.show()

        base_filename = os.path.splitext(filename)[0]
        np.save('{}_features.npy'.format(base_filename), batches)
        np.save('{}_rays.npy'.format(base_filename), total_ray_d)
        np.save('{}_target.npy'.format(base_filename), exp_batchified)

        quit()

        return batches, total_ray_d, exp_batchified

    def calc_forces_from_batches(self, batches, force_samples, rays):
        """Calculate forces from batches based on force parameters for debugging purposes"""
        forces = np.empty(((len(batches) // force_samples), 3))
#            forces = np.empty(len(batches) * (len(batches[0]) // feed_samples))
        force_idx = 0
        for kdx in tqdm(range(0, len(batches), force_samples)):
            force = np.zeros(3)
            #force = 0
            for jdx in range(force_samples):
                ray = batches[kdx + jdx]
                ray_info = rays[kdx + jdx]
                f_c = ((-1) * ray_info[:3]
                       * self.force_params[0] * ray[1]
                       * math.pow(ray[0], (1 - self.force_params[1])))
                f_n = ((-1) * ray_info[3:6]
                       * self.force_params[2] * ray[1]
                       * math.pow(ray[0], (1 - self.force_params[3])))
                f_t = ((-1) * ray_info[6:]
                       * self.force_params[4] * ray[1]
                       * math.pow(ray[0], (1 - self.force_params[5])))
                force += (f_c + f_n + f_t)
                    #force += ray[0]
            forces[force_idx] = force
            force_idx += 1
        return forces

    def predict(self, data_inp, data_out, rays, evaluation):
        """Capsuled prediction method using data and evaluation method"""
        total_error = 0
        total_pred = []
        for batch_idx, __ in enumerate(data_inp):
            inp, out = data_inp[batch_idx], data_out[batch_idx]
            pred_out, error = evaluation(inp, out, rays, batch_idx)
            total_error += error

            pred_npy = pred_out.cpu().numpy()

            for value in pred_npy[:, self.sync_axis]:
                total_pred.append(value)

        total_error /= len(data_inp)

        return total_pred, total_error

    def validate(self, evaluation, __, epoch_id):
        """Validation and visualization"""
        total_errors = np.empty(len(self.val_files))
        print("Start validation...")
        for idx in tqdm(range(len(self.val_files))):
            x__, y__, rays = self.prepare_batches(self.val_files[idx], self.val_numbers[idx])

            reference = np.reshape(y__, (-1, self.target_output_size))

            total_pred, total_error = self.predict(
                x__,
                y__,
                rays,
                evaluation
            )

            to_save = np.array([total_pred, reference[:, self.sync_axis]])
            save_str = "{}/number{}_epoch{}_error{}".format(
                self.config['results_dir'],
                self.val_numbers[idx],
                epoch_id,
                total_error
            )
            tdms.write_tdms(to_save, ['pred', 'ref'], 'Recorder', save_str + ".tdms")

            total_errors[idx] = total_error
            # if epoch_id != -1:
                # plot(
                    # [
                        # total_pred[len(total_pred) // 2:len(total_pred) // 2 + 100],
                        # reference[len(reference) // 2:len(reference) // 2 + 100, self.sync_axis]
                    # ],
                    # ['Prediction', 'Reference'],
                    # save_str + ".png"
                # )
        return total_errors.mean(), total_errors.std()

    def infer(self, evaluation):
        """Do predictions using learned model"""
        print("Start inference...")
        infer_files = np.concatenate((self.train_files, self.val_files))
        infer_numbers = np.concatenate((self.train_numbers, self.val_numbers))
        total_errors = np.empty(len(self.train_files))
        for idx in tqdm(range(len(infer_files))):
            infer_x = np.load("{}_features.npy".format(infer_files[idx]))
            infer_rays = np.load("{}_rays.npy".format(infer_files[idx]))
            infer_y = np.load("{}_target.npy".format(infer_files[idx]))

            # stepx = 2 * self.parameter_values[infer_numbers[idx]][-1]
            # batch_size = int(self.config['batched_turnarounds'] * stepx * self.feed_samples)
            batch_size = self.batch_size
            out_batch_size = batch_size // self.force_samples

            infer_x = np.reshape(
                self.x_scaler.transform(
                    infer_x[:(len(infer_x) // batch_size) * batch_size]
                ),
                (-1, batch_size, np.shape(infer_x)[-1])
            )
            infer_y = np.reshape(
                self.y_scaler.transform(
                    infer_y[:(len(infer_y) // out_batch_size) * out_batch_size]
                ),
                (-1, out_batch_size, np.shape(infer_y)[-1])
            )

            reference = np.reshape(infer_y, (-1, np.shape(infer_y)[-1]))

            total_pred, total_error = self.predict(
                infer_x,
                infer_y,
                infer_rays,
                evaluation
            )

            to_save = np.array([total_pred, reference[:, self.sync_axis]])
            save_str = "{}/{}_number{}_error{}".format(
                self.config['results_dir'],
                "train" if idx < len(self.train_files) else "val",
                infer_numbers[idx],
                total_error
            )
            tdms.write_tdms(to_save, ['pred', 'ref'], 'Recorder', save_str + ".tdms")

            total_errors[idx % len(self.train_files)] = total_error
            # plot(
                # [
                    # total_pred[len(total_pred) // 2:len(total_pred) // 2 + 100],
                    # reference[len(reference) // 2:len(reference) // 2 + 100, self.sync_axis]
                # ],
                # ['Prediction', 'Reference'],
                # save_str + ".png"
            # )
            if idx == len(self.train_files) - 1:
                print("Training error: {} +- {}".format(total_errors.mean(), total_errors.std()))
                total_errors = np.empty(len(self.val_files))
        print("Validation error: {} +- {}".format(total_errors.mean(), total_errors.std()))
