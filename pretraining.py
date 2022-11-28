import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models import CornYieldModel


class CornDataLoader(Dataset):
    def __init__(self, features, target, sequence_length=365):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length

    def __len__(self):
        return self.target.shape[0] * self.target.shape[1]

    def __getitem__(self, i):
        i_loc = i // self.target.shape[1]
        i_year = i % self.target.shape[1]

        x = self.features[i_loc, i_year * 365:(i_year + 1) * 365, :]
        y = self.target[i_loc, i_year, :]

        return x, y


class R2Loss(nn.Module):
    # calculate coefficient of determination
    def forward(self, y_pred, y):
        var_y = torch.var(y, unbiased=False)
        return 1.0 - F.mse_loss(y_pred, y, reduction="mean") / var_y


def my_loss(output, target):
    loss = torch.mean((output - target) ** 2)
    return loss


def myloss_mul_sum(output, target, loss_weights):
    loss = 0.0
    nout = output.size(2)
    for i in range(nout):
        loss = loss + loss_weights[i] * torch.mean((output[:, :, i] - target[:, :, i]) ** 2)
    return loss


def Z_norm(X):
    X_mean = X.numpy().mean(dtype=np.float64)
    X_std = np.std(np.array(X, dtype=np.float64))
    return (X - X_mean) / X_std, X_mean, X_std


def Z_norm_reverse(X, Xscaler, units_convert=1.0):
    return (X * Xscaler[1] + Xscaler[0]) * units_convert


def scalar_maxmin(X):
    return (X - X.min()) / (X.max() - X.min())


def Z_norm_with_scaler(X, Xscaler):
    return (X - Xscaler[0]) / Xscaler[1]


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    output_all = torch.tensor([]).to(device)
    y_all = torch.tensor([]).to(device)

    for X, y in data_loader:
        output = model(X.to(device)).squeeze(2)
        loss = loss_function(output, y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        output_all = torch.cat((output_all, output), 0)
        y_all = torch.cat((y_all, y.to(device)), 0)

    avg_loss = total_loss / num_batches
    R2 = compute_r2(output_all, y_all).detach().cpu().numpy()

    print(f"Train loss: {avg_loss}", "R2:", R2)
    return avg_loss, R2


def evaluate_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0
    model.eval()

    output_all = torch.tensor([]).to(device)
    y_all = torch.tensor([]).to(device)

    with torch.no_grad():
        for X, y in data_loader:
            output = model(X.to(device)).squeeze(2)
            total_loss += loss_function(output, y.to(device)).item()

            output_all = torch.cat((output_all, output), 0)
            y_all = torch.cat((y_all, y.to(device)), 0)

    avg_loss = total_loss / num_batches
    R2 = compute_r2(output_all, y_all).cpu().numpy()

    print(f"Validation loss: {avg_loss}", "R2:", R2)
    return avg_loss, R2


def predict(data_loader, model):
    output = torch.tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_pred = model(X.to(device)).squeeze(2)
            output = torch.cat((output, y_pred), 0)

    return output


def preprocess_data(split="crop"):
    base_path = './data/'
    # load county ids
    county_FIPS = np.load(os.path.join(base_path, 'county_FIPS.npy')).tolist()
    dropsites = [38039, 27123, 47095, 47183, 47053, 47045,
                 47079]  # remove the county without Corn/Soybean rotation fields
    for s in dropsites:
        county_FIPS.remove(s)

    # load corn and soybean feature dictions {county_id: features}
    # features have 300*21 dimension -- 300 sample points collected in 21 years
    corn_fraction = np.load(os.path.join(base_path, 'corn_fraction_sample_300.npy'), allow_pickle=True).item()
    soybean_fraction = np.load(os.path.join(base_path, 'soybean_fraction_sample_300.npy'), allow_pickle=True).item()

    # load corn and soybean target dictions {county_id: targets}
    # targets have 21*1 dimensions -- yeld values in 21 years
    obs_corn_yield = np.load(os.path.join(base_path, 'obs_corn_yield.npy'), allow_pickle=True).item()
    obs_soybean_yield = np.load(os.path.join(base_path, 'obs_soybean_yield.npy'), allow_pickle=True).item()

    # load coefficience that convert gC/m2/year to Bu/Acre
    convert_index_corn = obs_corn_yield['gC/m2/year_to_Bu/Acre']
    convert_index_soybean = obs_soybean_yield['gC/m2/year_to_Bu/Acre']

    print("cleaned county size", len(county_FIPS))

    print("corn fraction shape", corn_fraction[county_FIPS[0]].shape)
    print("corn yield shape", obs_corn_yield[county_FIPS[0]].shape)

    print("soybean fraction shape", soybean_fraction[county_FIPS[0]].shape)
    print("soybean yield shape", obs_soybean_yield[county_FIPS[0]].shape)

    ################### parameter configuration

    data_path = base_path + 'recotest_data_scaled_v4.sav'

    start = 2001
    end = 2018
    Tx = 365  # timesteps
    tyear = end - start + 1

    out1_names = ['Ra', 'Rh', 'NEE']
    n_out1 = len(out1_names)

    out2_names = ['Yield']
    n_out2 = len(out2_names)

    # time series data name
    fts_names = ['RADN', 'TMAX_AIR', 'TDIF_AIR', 'HMAX_AIR', 'HDIF_AIR', 'WIND', 'PRECN', 'Crop_Type', 'GPP', 'Ra', 'Rh', 'GrainC']
    # SP data name
    fsp_names = ['TBKDS', 'TSAND', 'TSILT', 'TFC', 'TWP', 'TKSat', 'TSOC', 'TPH', 'TCEC']
    # toatal fiedls (features) but no Ra Rh GrainC
    f_names = ['RADN', 'TMAX_AIR', 'TDIF_AIR', 'HMAX_AIR', 'HDIF_AIR', 'WIND', 'PRECN', 'Crop_Type', 'GPP'] + ['Year'] + fsp_names
    print("feature names", f_names)

    # load data recotest_data_scaled_v4.sav
    data0 = torch.load(data_path)

    # FIPS reference id
    FIPS_ref = data0['FIPS_ref']
    print("FIPS ref shape/batch size", FIPS_ref.shape)

    # batch size? what does it mean?
    bsz0 = len(FIPS_ref)
    n_f = len(f_names)

    #################### initialize X, Y1, and Y2

    # initial input X (days, batch size, num of features)
    # X_scaler (mean, std) of X
    X = torch.zeros([Tx * tyear, bsz0, n_f])
    X_scaler = np.zeros([n_f, 2])
    print("feature X shape", X.shape, "X scaler shape", X_scaler.shape)

    # initial output Y1 (days, batch size, num of targets)
    Y1 = torch.zeros([Tx * tyear, bsz0, n_out1])
    Y1_scaler = np.zeros([n_out1, 2])
    print("target Y1 shape", Y1.shape, "Y1 scaler shape", Y1_scaler.shape)

    # initial output Y2 (year, batch size, num of targets)
    Y2 = torch.zeros([tyear, bsz0, n_out2])
    Y2_scaler = np.zeros([n_out2, 2])
    print("target Y2 shape", Y2.shape, "Y2 scaler shape", Y2_scaler.shape)

    #################### load in X, Y1, and Y2 from recotest_data_scaled_v4.sav
    # load in X for the first 9 features
    X[:, :, 0:9] = data0['X'][:, :, 0:9]
    X_scaler[0:9, :] = data0['X_scaler'][0:9, :]

    # load in X for the 9th feature - year
    for y in range(tyear):
        X[y * Tx:(y + 1) * Tx, :, 9] = y + start
    X[:, :, 9], X_scaler[9, 0], X_scaler[9, 1] = Z_norm(X[:, :, 9])

    # load the rest 9 features
    for i in range(len(fsp_names)):
        X[:, :, 10 + i] = data0['Xsp'][:, i].view(1, bsz0).repeat(Tx * tyear, 1)
        X_scaler[10 + i, :] = data0['Xsp_scaler'][i, :]

    ###################

    # load in Y1
    Y1_scaler[0:2, :] = data0['X_scaler'][9:11, :]
    for i in range(2):
        Y1[:, :, i] = Z_norm_reverse(data0['X'][:, :, 9 + i], Y1_scaler[i, :], 1.0)

    # remove points over 0
    # Y1[Y1>0.0] = 0.0
    GPP = Z_norm_reverse(X[:, :, 8], X_scaler[8, :], 1.0)
    # GPP -Ra-Rh, Ra, Ra are negative,GPP +Ra+Rh+NEE =0
    Y1[:, :, 2] = -(GPP + Y1[:, :, 0] + Y1[:, :, 1])
    for i in range(3):
        Y1[:, :, i], Y1_scaler[i, 0], Y1_scaler[i, 1] = Z_norm(Y1[:, :, i])

    ####################

    # load in Y2
    Y2_scaler[:, :] = data0['X_scaler'][11, :]
    for y in range(tyear):
        Y2[y, :, 0] = Z_norm_reverse(data0['X'][(y + 1) * Tx - 2, :, 11], Y2_scaler[0, :], 1.0)
    Y2[:, :, 0], Y2_scaler[0, 0], Y2_scaler[0, 1] = Z_norm(Y2[:, :, 0])

    ####################

    # calculate the fraction of Res to GPP
    GPP_annual_all = torch.zeros([tyear, bsz0])
    Ra_annual_all = torch.zeros([tyear, bsz0])
    for y in range(tyear):
        GPP_annual_all[y, :] = torch.sum(Z_norm_reverse(X[y * Tx:(y + 1) * Tx, :, 8], X_scaler[8, :], 1.0), dim=0)
        Ra_annual_all[y, :] = torch.sum(Z_norm_reverse(Y1[y * Tx:(y + 1) * Tx, :, 0], Y1_scaler[0, :], 1.0), dim=0)
    Res_annual_all = GPP_annual_all + Ra_annual_all - Z_norm_reverse(Y2[:, :, 0], Y2_scaler[0, :], 1.0)
    GPP_Res_f = torch.mean(GPP_annual_all, dim=0) / torch.mean(Res_annual_all, dim=0)
    GPP_Res_fmean = GPP_Res_f.mean()
    Res_scaler = np.zeros([1, 2])

    # feature scaling of Res
    Res__, Res_scaler[0, 0], Res_scaler[0, 1] = Z_norm(Res_annual_all)
    print("GPP_Res_fmean and Res_scaler", GPP_Res_fmean, Res_scaler)

    X_new = torch.from_numpy(np.einsum('ijk->jik', X.numpy()))
    Y2_new = torch.from_numpy(np.einsum('ijk->jik', Y2.numpy()))

    if split == "year":
        #####################################################
        # split the X into 3 sets: first 10 years as train;
        # the following 4 years as validation;
        # the rest 4 years as test
        split1, split2 = 10, 14
        X_train_data = X_new[:, :split1 * 365, :]
        X_valid_data = X_new[:, split1 * 365:split2 * 365, :]
        X_test_data = X_new[:, split2 * 365:, :]

        Y_train_data = Y2_new[:, :split1, :]
        Y_valid_data = Y2_new[:, split1:split2, :]
        Y_test_data = Y2_new[:, split2:, :]
    else:
        #####################################################
        # split the X into 3 sets: first 60% local data as train;
        # the following 20% local data as validation;
        # the rest 20% local data as test
        split1, split2 = int(0.6 * 10335), int(0.8 * 10335)
        X_train_data = X_new[:split1, :, :]
        X_valid_data = X_new[split1:split2, :, :]
        X_test_data = X_new[split2:, :, :]

        Y_train_data = Y2_new[:split1, :, :]
        Y_valid_data = Y2_new[split1:split2, :, :]
        Y_test_data = Y2_new[split2:, :, :]

    ####################################################

    print("X train", X_train_data.shape)
    print("Y train", Y_train_data.shape)

    print("X valid", X_valid_data.shape)
    print("Y valid", Y_valid_data.shape)

    print("X_test", X_test_data.shape)
    print("Y_test", Y_test_data.shape)

    return X_train_data, X_valid_data, X_test_data, Y_train_data, Y_valid_data, Y_test_data, Y2_scaler


if __name__ == '__main__':

    X_train_data, X_valid_data, X_test_data, Y_train_data, Y_valid_data, Y_test_data, Y2_scaler = preprocess_data(
        split="crop")

    train_dataset = CornDataLoader(
        X_train_data,
        Y_train_data,
        sequence_length=365
    )

    valid_dataset = CornDataLoader(
        X_valid_data,
        Y_valid_data,
        sequence_length=365
    )

    test_dataset = CornDataLoader(
        X_test_data,
        Y_test_data,
        sequence_length=365
    )

    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4096, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

    # load gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"{device} is available!")

    model = CornYieldModel(num_features=19, hidden_size=64).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    compute_r2 = R2Loss()

    best_loss = 10000
    model_name = "best_model_synthetic_split_by_space_v2.pkl"

    for epoch in range(4):
        print(f"Epoch {epoch}\n---------")
        train_loss, train_R2 = train_model(train_loader, model, loss_function, optimizer=optimizer)
        valid_loss, valid_R2 = evaluate_model(valid_loader, model, loss_function)

        if valid_loss < best_loss:
            best_epoch = epoch
            torch.save({'epoch': best_epoch,
                        'model_state_dict': model.state_dict(),
                        'train_loss': train_loss,
                        'train_R2': train_R2,
                        'valid_loss': valid_loss,
                        'valid_R2': valid_R2,
                        }, f"../checkpoints/{model_name}")

    # load the best model
    model_name = "best_model_synthetic_split_by_time.pkl"
    checkpoint = torch.load(f"./checkpoints/{model_name}")
    model = CornYieldModel(num_features=19, hidden_size=64).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    Y_test_pred = predict(test_loader, model).cpu()
    Y_test_gold = Y_test_data.squeeze(2).reshape((-1, 1))

    Y_test_pred = Z_norm_reverse(Y_test_pred, Y2_scaler[0])
    Y_test_gold = Z_norm_reverse(Y_test_gold, Y2_scaler[0])

    print("Y_test_pred shape", Y_test_pred.shape)
    print("Y_test_gold shape", Y_test_pred.shape)

    compute_r2 = R2Loss()
    R2 = compute_r2(Y_test_gold, Y_test_pred).numpy()
    RMSE = np.sqrt(my_loss(Y_test_gold, Y_test_pred).numpy())
    Bias = torch.mean(Y_test_gold - Y_test_pred).numpy()
    slop, intercept, r_value, p_value, std_err = stats.linregress(Y_test_gold.contiguous().view(-1).numpy(),
                                                                  Y_test_pred.contiguous().view(-1).numpy())
    # plot figures

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(Y_test_pred.contiguous().view(-1).numpy(), Y_test_gold.contiguous().view(-1).numpy(), s=1, color='black',
               alpha=0.5)
    ax.plot([0, 600], [0, 600], color='red', linestyle='--')

    print("R2", R2)
    print("RMSE", RMSE)
    print("Bias", Bias)

    # load the best model
    model_name = "best_model_synthetic_split_by_space.pkl"
    checkpoint = torch.load(f"./checkpoints/{model_name}")
    model = CornYieldModel(num_features=19, hidden_size=64).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    Y_test_pred = predict(test_loader, model).cpu()
    Y_test_gold = Y_test_data.squeeze(2).reshape((-1, 1))

    Y_test_pred = Z_norm_reverse(Y_test_pred, Y2_scaler[0])
    Y_test_gold = Z_norm_reverse(Y_test_gold, Y2_scaler[0])

    print("Y_test_pred shape", Y_test_pred.shape)
    print("Y_test_gold shape", Y_test_pred.shape)

    compute_r2 = R2Loss()
    R2 = compute_r2(Y_test_gold, Y_test_pred).numpy()
    RMSE = np.sqrt(my_loss(Y_test_gold, Y_test_pred).numpy())
    Bias = torch.mean(Y_test_gold - Y_test_pred).numpy()
    slop, intercept, r_value, p_value, std_err = stats.linregress(Y_test_gold.contiguous().view(-1).numpy(),
                                                                  Y_test_pred.contiguous().view(-1).numpy())
    # plot figures

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(Y_test_pred.contiguous().view(-1).numpy(), Y_test_gold.contiguous().view(-1).numpy(), s=1, color='black',
               alpha=0.5)
    ax.plot([0, 600], [0, 600], color='red', linestyle='--')

    print("R2", R2)
    print("RMSE", RMSE)
    print("Bias", Bias)
