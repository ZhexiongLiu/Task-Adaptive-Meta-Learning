import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        # x = x.view(-1, x.size(1))
        x = x.reshape(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []

        hn = h0[0, :, :]

        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:, seq, :], hn)
            outs.append(hn)

        # out = outs[-1].squeeze()
        out = torch.stack(outs, dim=1)

        out = self.fc(out)
        return out


class CornYieldModel(nn.Module):
    def __init__(self, num_features, hidden_size=365, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru_basic = GRUModel(num_features,
                                  hidden_size,
                                  num_layers,
                                  64)

        self.drop = nn.Dropout(dropout)

        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)

        self.attn = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        self.densor_yield = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.ReLU = nn.ReLU()

    def forward(self, x):
        # output, _ = self.gru_basic(x)
        output = self.gru_basic(x)
        inputs2 = self.drop(output)
        attn_weights = F.softmax(self.attn(inputs2), dim=1).view(x.size(0), 1, x.size(1))
        inputs2 = torch.bmm(attn_weights, inputs2)
        output2 = self.densor_yield(inputs2)

        return output2


class LossFunctions(object):

    def __init__(self, is_mean_loss=False):
        self.is_mean_loss = is_mean_loss
        self.mse_loss = nn.MSELoss()

    def mean_mse_loss_func_v1(self, y_pred, corn_mask, soybean_mask, corn_y_gold, soybean_y_gold, fips_id, year_id):
        corn_pred_dict = {}
        soybean_pred_dict = {}
        corn_gold_dict = {}
        soybean_gold_dict = {}

        corn_pred = torch.masked_select(y_pred, corn_mask)
        soybean_pred = torch.masked_select(y_pred, soybean_mask)

        for j in range(corn_pred.shape[0]):
            dict_key = (fips_id.cpu().numpy()[j], year_id.cpu().numpy()[j])
            if dict_key not in corn_pred_dict:
                corn_pred_dict[dict_key] = [[corn_pred.cpu()[j]]]
            else:
                corn_pred_dict[dict_key] += [[corn_pred.cpu()[j]]]

            if dict_key not in corn_gold_dict:
                corn_gold_dict[dict_key] = [corn_y_gold[j]]
            else:
                corn_gold_dict[dict_key] += [corn_y_gold[j]]

        for j in range(soybean_pred.shape[0]):
            dict_key = (fips_id.cpu().numpy()[j], year_id.cpu().numpy()[j])
            if dict_key not in soybean_pred_dict:
                soybean_pred_dict[dict_key] = [[soybean_pred.cpu()[j]]]
            else:
                soybean_pred_dict[dict_key] += [[soybean_pred.cpu()[j]]]

            if dict_key not in soybean_gold_dict:
                soybean_gold_dict[dict_key] = [soybean_y_gold[j]]
            else:
                soybean_gold_dict[dict_key] += [soybean_y_gold[j]]

        pred_corn = []
        pred_soybean = []
        gold_corn = []
        gold_soybean = []

        corn_pred_mean = {key: torch.mean(torch.FloatTensor(corn_pred_dict[key])) for key in corn_pred_dict}
        corn_gold_mean = {key: torch.mean(torch.FloatTensor(corn_gold_dict[key])) for key in corn_gold_dict}
        soybean_pred_mean = {key: torch.mean(torch.FloatTensor(soybean_pred_dict[key])) for key in soybean_pred_dict}
        soybean_gold_mean = {key: torch.mean(torch.FloatTensor(soybean_gold_dict[key])) for key in soybean_gold_dict}

        for key in corn_pred_mean:
            pred_corn.append(corn_pred_mean[key])
            gold_corn.append(corn_gold_mean[key])

        for key in soybean_pred_mean:
            pred_soybean.append(soybean_pred_mean[key])
            gold_soybean.append(soybean_gold_mean[key])

        pred_corn = torch.FloatTensor(pred_corn)
        pred_soybean = torch.FloatTensor(pred_soybean)
        gold_corn = torch.FloatTensor(gold_corn)
        gold_soybean = torch.FloatTensor(gold_soybean)

        loss_corn = self.mse_loss(gold_corn, pred_corn)
        loss_soybean = self.mse_loss(gold_soybean, pred_soybean)

        return loss_corn + loss_soybean

    def mean_mse_loss_func(self, y_pred, corn_mask, soybean_mask, corn_y_gold, soybean_y_gold, fips_id, year_id):
        corn_mask = corn_mask.unsqueeze(1)
        soybean_mask = soybean_mask.unsqueeze(1)
        if len(corn_y_gold.shape) == 1:
            corn_y_gold = corn_y_gold.unsqueeze(1)
        if len(soybean_y_gold.shape) == 1:
            soybean_y_gold = soybean_y_gold.unsqueeze(1)
        fips_id = fips_id.unsqueeze(1)
        year_id = year_id.unsqueeze(1)

        corn_pred = torch.masked_select(y_pred, corn_mask)
        corn_y_gold = torch.masked_select(corn_y_gold, corn_mask)
        corn_fips_id = torch.masked_select(fips_id, corn_mask)
        corn_year_id = torch.masked_select(year_id, corn_mask)

        soybean_pred = torch.masked_select(y_pred, soybean_mask)
        soybean_y_gold = torch.masked_select(soybean_y_gold, soybean_mask)
        soybean_fips_id = torch.masked_select(fips_id, soybean_mask)
        soybean_year_id = torch.masked_select(year_id, soybean_mask)

        # corn_fips_id+corn_year_id*0.1 is a unique id for each county in a year
        corn_samples = torch.stack(
            [corn_pred, corn_y_gold, corn_fips_id, corn_year_id, corn_fips_id * 100 + corn_year_id],
            1)
        corn_labels = corn_fips_id * 100 + corn_year_id
        corn_mean_pred, corn_mean_gold, corn_mean_fips_id, corn_mean_year_id = self.group_mean(corn_samples,
                                                                                               corn_labels)

        # soybean_fips_id+soybean_year_id*0.1 is a unique id for each county in a year
        soybean_samples = torch.stack(
            [soybean_pred, soybean_y_gold, soybean_fips_id, soybean_year_id, soybean_fips_id * 100 + soybean_year_id],
            1)
        soybean_labels = soybean_fips_id * 100 + soybean_year_id
        soybean_mean_pred, soybean_mean_gold, soybean_mean_fips_id, soybean_mean_year_id = self.group_mean(
            soybean_samples,
            soybean_labels)

        if len(corn_mean_pred) > 0 and len(soybean_mean_pred) > 0:
            loss_corn = self.mse_loss(corn_mean_pred, corn_mean_gold)
            loss_soybean = self.mse_loss(soybean_mean_pred, soybean_mean_gold)
            loss = loss_corn + loss_soybean
        elif len(corn_mean_pred) > 0:
            loss = self.mse_loss(corn_mean_pred, corn_mean_gold)
        elif len(soybean_mean_pred) > 0:
            loss = self.mse_loss(soybean_mean_pred, soybean_mean_gold)
        else:
            loss = torch.FloatTensor(0)

        return loss

    def group_mean(self, samples, labels):
        '''
        select mean(samples)from samples group by labels order by labels asc
        '''

        if len(samples) == 0:
            return labels, labels, labels, labels

        labels = (labels - labels.min())
        row_num = (labels.max() + 1).cpu().tolist()
        weight = torch.zeros(row_num, samples.shape[0]).to(samples.device)  # L, N
        weight[labels, torch.arange(samples.shape[0])] = 1
        label_count = weight.sum(dim=1)
        weight = torch.nn.functional.normalize(weight, p=1, dim=1)  # l1 normalization
        mean = torch.mm(weight, samples)  # L, F
        index = torch.arange(mean.shape[0])[label_count > 0]
        # return mean[index], label_count[index]

        predict = mean[index][:, 0]
        gold = mean[index][:, 1]
        fips_id = mean[index][:, 2]
        year_id = mean[index][:, 3]

        return predict, gold, fips_id, year_id

    def mse_loss_func(self):
        return self.mse_loss
