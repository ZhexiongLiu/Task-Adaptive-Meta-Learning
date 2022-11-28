import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models import LossFunctions


class R2Loss(nn.Module):
    def forward(self, y_pred, y):
        var_y = torch.var(y, unbiased=False)
        return 1.0 - F.mse_loss(y_pred, y, reduction="mean") / var_y


class HelperFunctions(object):

    def __init__(self):
        self.data_path = './meta/data/'
        self.project_path = './meta/'
        self.dropsites = [38039, 27123, 47095, 47183, 47053, 47045, 47079]
        self.scalar = torch.load(os.path.join(self.data_path, 'recotest_data_scaled_v4_scalers.sav'))['Y2_scaler']

    def Z_norm(self, X):
        X_mean = X.numpy().mean(dtype=np.float64)
        X_std = np.std(np.array(X, dtype=np.float64))
        return (X - X_mean) / X_std, X_mean, X_std

    def Z_norm_reverse(self, X, Xscaler, units_convert=1.0):
        return (X * Xscaler[1] + Xscaler[0]) * units_convert

    def scalar_maxmin(self, X):
        return (X - X.min()) / (X.max() - X.min())

    def Z_norm_with_scaler(self, X, Xscaler):
        return (X - Xscaler[0]) / Xscaler[1]

    def pad_array(self, arr, size):
        # support at most 3-d array
        shape = np.shape(arr)
        res = np.zeros(size)
        if len(shape) == 2:
            res[:shape[0], :shape[1]] = arr
        elif len(shape) == 3:
            res[:shape[0], :shape[1], :shape[2]] = arr
        else:
            res[:shape[0]] = arr
        return res

    def load_raw_target_data(self):

        # load the scaler
        data_path = self.data_path
        scalar = self.scalar

        # remove the county without Corn/Soybean rotation fields

        county_FIPS = np.load(data_path + 'county_FIPS.npy')
        county_FIPS = [i for i in county_FIPS if i not in self.dropsites]
        county_FIPS = np.array(county_FIPS)

        # load corn and soybean fraction (Dictionary)
        corn_fraction = np.load(os.path.join(data_path, 'corn_fraction_sample_300.npy'), allow_pickle=True).item()
        soybean_fraction = np.load(os.path.join(data_path, 'soybean_fraction_sample_300.npy'), allow_pickle=True).item()

        # load observed crop yields (Dictionary)
        obs_corn_yield = np.load(os.path.join(data_path, 'obs_corn_yield.npy'), allow_pickle=True).item()
        obs_soybean_yield = np.load(os.path.join(data_path, 'obs_soybean_yield.npy'), allow_pickle=True).item()

        Y_corn_new = []
        Y_corn_fraction_new = []
        Y_soybean_new = []
        Y_soybean_fraction_new = []

        for county_id in county_FIPS:
            if corn_fraction[county_id].shape != (300, 21):
                corn_fraction[county_id] = self.pad_array(corn_fraction[county_id], (300, 21))
            if soybean_fraction[county_id].shape != (300, 21):
                soybean_fraction[county_id] = self.pad_array(soybean_fraction[county_id], (300, 21))
            Y_corn_new.append(obs_corn_yield[county_id].tolist())
            Y_corn_fraction_new.append(corn_fraction[county_id])
            Y_soybean_new.append(obs_soybean_yield[county_id].tolist())
            Y_soybean_fraction_new.append(soybean_fraction[county_id])

        Y_corn_new = np.array(Y_corn_new)
        Y_corn_fraction_new = np.stack(Y_corn_fraction_new, axis=0)
        Y_soybean_new = np.array(Y_soybean_new)
        Y_soybean_fraction_new = np.stack(Y_soybean_fraction_new, axis=0)

        # # expand dimension
        if len(Y_corn_new.shape) < 3:
            Y_corn_new = np.expand_dims(Y_corn_new, axis=2)
            Y_soybean_new = np.expand_dims(Y_soybean_new, axis=2)

        print("cleaned county size", len(county_FIPS))
        print("Y corn fraction shape", Y_corn_fraction_new.shape)
        print("Y corn yields shape", Y_corn_new.shape)
        print("Y soybean fraction shape", Y_soybean_fraction_new.shape)
        print("Y soybean yields shape", Y_soybean_new.shape)

        return county_FIPS, Y_corn_fraction_new, Y_soybean_fraction_new, Y_corn_new, Y_soybean_new

    def load_subsample_data(self, sample_size=15):
        # store cache data
        cache_path = os.path.join(self.project_path, "cache_data_subsample.pickle")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                data, ylds, corn_frac, soybean_frac, corn_ylds, soybean_ylds, yld_ids = cache
            return data, ylds, corn_frac, soybean_frac, corn_ylds, soybean_ylds, yld_ids

        data = []
        ylds = []
        corn_frac = []
        soybean_frac = []
        corn_ylds = []
        soybean_ylds = []
        # a combination of county id and year as a label
        yld_ids = []

        county_FIPS, corn_fraction, soybean_fraction, corn_yields, soybean_yields = self.load_raw_target_data()

        # load the county basic data
        data_path = os.path.join(self.data_path, 'combine_random_sample_size_300/')

        for i in tqdm(range(len(county_FIPS))):
            fips = county_FIPS[i]
            # remove county that has nan features
            if fips in self.dropsites:
                continue
            data_i = np.load(os.path.join(data_path, f'Pred_xset_{fips}_sample_300.npy.npz'))['arr_0']

            if data_i.shape != (300, 7665, 19):
                data_i = self.pad_array(data_i, (300, 7665, 19))

            random_indices = np.random.choice(300, size=sample_size, replace=False)
            data_tmp = data_i[random_indices, :, :]
            corn_fraction_tmp = corn_fraction[i, random_indices, :]
            soybean_fraction_tmp = soybean_fraction[i, random_indices, :]

            if np.isnan(data_tmp).any():
                print(f"nan in feature data {fips}")

            data.append(data_tmp)

            for k in range(sample_size):
                corn_fraction_k = corn_fraction_tmp[k, :]
                soybean_fraction_k = soybean_fraction_tmp[k, :]
                corn_fraction_k = np.expand_dims(corn_fraction_k, axis=1)
                soybean_fraction_k = np.expand_dims(soybean_fraction_k, axis=1)

                corn_frac.append(corn_fraction_k)
                soybean_frac.append(soybean_fraction_k)

                yield_tmp = corn_yields[i] * (corn_fraction_k > 0.5) + soybean_yields[i] * (soybean_fraction_k > 0.5)

                if np.any(yield_tmp):
                    print(f"yields has 0 values {fips} at sample {k}")
                    # print(yield_tmp)

                if np.isnan(yield_tmp).any():
                    print(f"yields has nan values {fips} at sample {k}")

                ylds.append(yield_tmp)
                corn_ylds.append(corn_yields[i])
                soybean_ylds.append(soybean_yields[i])

                # county, location, year
                for j in range(corn_yields[i].shape[0]):
                    yld_ids.append((fips, k, j))

        data = np.concatenate(data, axis=0)
        ylds = np.array(ylds)
        corn_ylds = np.array(corn_ylds)
        soybean_ylds = np.array(soybean_ylds)
        yld_ids = np.array(yld_ids)

        corn_frac = np.array(corn_frac)
        soybean_frac = np.array(soybean_frac)

        data = np.nan_to_num(data)
        ylds = np.nan_to_num(ylds)
        corn_ylds = np.nan_to_num(corn_ylds)
        soybean_ylds = np.nan_to_num(soybean_ylds)

        ylds = self.Z_norm_with_scaler(ylds, self.scalar[0])
        corn_ylds = self.Z_norm_with_scaler(corn_ylds, self.scalar[0])
        soybean_ylds = self.Z_norm_with_scaler(soybean_ylds, self.scalar[0])

        ylds = torch.from_numpy(np.float32(ylds))
        corn_ylds = torch.from_numpy(np.float32(corn_ylds))
        soybean_ylds = torch.from_numpy(np.float32(soybean_ylds))

        if not os.path.exists(cache_path):
            cache = (data, ylds, corn_frac, soybean_frac, corn_ylds, soybean_ylds, yld_ids)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f)

        return data, ylds, corn_frac, soybean_frac, corn_ylds, soybean_ylds, yld_ids

    def load_mean_data(self, corn_yield, soybean_yield):

        cache_path = os.path.join(self.project_path, "cache_data_mean.pickle")

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                data, ylds, corn_frac, soybean_frac, corn_ylds, soybean_ylds, yld_ids = cache
            return data, ylds, corn_frac, soybean_frac, corn_ylds, soybean_ylds, yld_ids

        data = []
        ylds = []
        corn_frac = []
        soybean_frac = []
        corn_ylds = []
        soybean_ylds = []
        # a combination of county id and year as a label
        yld_ids = []

        county_FIPS, corn_fraction, soybean_fraction, corn_yields, soybean_yields = self.load_raw_target_data()

        # load the county basic data
        data_path = os.path.join(self.data_path, 'combine_random_sample_size_300/')

        for i in tqdm(range(len(county_FIPS))):
            fips = county_FIPS[i]
            # remove county that has nan features
            if fips in self.dropsites:
                continue
            data_i = np.load(os.path.join(data_path, f'Pred_xset_{fips}_sample_300.npy.npz'))['arr_0']

            if data_i.shape != (300, 7665, 19):
                data_i = self.pad_array(data_i, (300, 7665, 19))

            data.append(data_i.mean(0))
            corn_frac_mean = corn_fraction.mean(1)[i, :]
            soybean_frac_mean = soybean_fraction.mean(1)[i, :]

            corn_frac = np.expand_dims(corn_frac, axis=1)
            soybean_frac = np.expand_dims(soybean_frac, axis=1)

            yield_tmp = corn_yields[i] * (corn_frac_mean > 0.5) + soybean_yields[i] * (soybean_frac_mean > 0.5)

            ylds.append(yield_tmp)

            corn_frac.append(corn_frac_mean)
            soybean_frac.append(soybean_frac_mean)

            corn_ylds.append(corn_yields[i])
            soybean_ylds.append(soybean_yields[i])

            # county, location, year
            for j in range(corn_yields[i].shape[0]):
                yld_ids.append((fips, 0, j))

        data = np.concatenate(data, axis=0)
        ylds = np.array(ylds)
        corn_ylds = np.array(corn_ylds)
        soybean_ylds = np.array(soybean_ylds)
        yld_ids = np.array(yld_ids)

        corn_frac = np.array(corn_frac)
        soybean_frac = np.array(soybean_frac)

        data = np.nan_to_num(data)
        ylds = np.nan_to_num(ylds)
        corn_ylds = np.nan_to_num(corn_ylds)
        soybean_ylds = np.nan_to_num(soybean_ylds)

        ylds = self.Z_norm_with_scaler(ylds, self.scalar[0])
        corn_ylds = self.Z_norm_with_scaler(corn_ylds, self.scalar[0])
        soybean_ylds = self.Z_norm_with_scaler(soybean_ylds, self.scalar[0])

        ylds = torch.from_numpy(np.float32(ylds))
        corn_ylds = torch.from_numpy(np.float32(corn_ylds))
        soybean_ylds = torch.from_numpy(np.float32(soybean_ylds))

        if not os.path.exists(cache_path):
            cache = (data, ylds, corn_frac, soybean_frac, corn_ylds, soybean_ylds, yld_ids)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f)

        return data, ylds, corn_frac, soybean_frac, corn_ylds, soybean_ylds, yld_ids

    def plot(self, corn_pred, gold_corn, file_path, name, crop="corn"):
        # revert normalization

        mse_loss_func = LossFunctions().mse_loss_func()

        compute_r2 = R2Loss()
        R2 = compute_r2(corn_pred, gold_corn).detach().cpu().numpy()
        RMSE = np.sqrt(mse_loss_func(corn_pred, gold_corn).detach().cpu().numpy())
        Bias = torch.mean(corn_pred - gold_corn).detach().cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        ax.scatter(corn_pred.detach().cpu().tolist(), gold_corn.detach().cpu().tolist(),
                   s=1, color='black', alpha=0.5)

        print("R2", R2, "RMSE", RMSE, "Bias", Bias, "\n")
        # print("RMSE", RMSE)
        # print("Bias", Bias)

        if crop == "corn":
            ax.plot([-2, 600], [-2, 600], color='red', linestyle='--')
            ax.text(5, 520, 'R$^2$=%0.3f\nRMSE=%0.3f\nbias=%0.3f' % (R2, RMSE, Bias), fontsize=12)
        else:
            ax.plot([-2, 300], [-2, 300], color='red', linestyle='--')
            ax.text(5, 260, 'R$^2$=%0.3f\nRMSE=%0.3f\nbias=%0.3f' % (R2, RMSE, Bias), fontsize=12)
        ax.set_xlabel("predicted values")
        ax.set_ylabel("gold values")
        ax.set_title(name, fontsize=15, weight='bold')

        # plt.show()
        plt.savefig(f"{file_path}/{name}.png")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
