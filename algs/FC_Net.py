import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from typing import Dict
from typing import Tuple
import copy
import time
from algs.contact_data_loader import DataLoader
import numpy as np

class Net(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: List[int]):
        super(Net, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        
        dim = input_dim
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(dim, hdim))
            dim = hdim
        self.layers.append(nn.Linear(dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)
        return out

class BinaryGRFNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: List[int]):
        super(BinaryGRFNet, self).__init__()

        shared_layer_hidden_dims = [256]
        shared_layer_output_dim = 256
        self.shared_layers = Net(input_dim, shared_layer_output_dim, shared_layer_hidden_dims)

        self.grfc_head = Net(shared_layer_output_dim, output_dim, hidden_dim=[])
        self.bc_head = Net(shared_layer_output_dim, 2, hidden_dim=[])

        self.layer_norm1 = torch.nn.LayerNorm(shared_layer_output_dim)
    def forward(self, x):

        x = self.shared_layers(x)

        grfc = self.grfc_head(x)

        bc_head = torch.sigmoid(self.bc_head(x))

        grfc[:, 0:6]  *= bc_head[:,0].reshape(-1,1) >= .5
        grfc[:, 6:12] *= bc_head[:,1].reshape(-1,1) >= .5

        return grfc

class FC_Net:
    def __init__(self):
        self.num_epochs = 20000
        self.batch_size = 500
        self.xy_split = 12
        hidden_dim = [256, 256]

        self.data_loader = DataLoader('/home/takaraet/fc_modeling/grfc_data.npy')

        train_output_mean = self.data_loader.train_mean[-self.xy_split:]
        self.train_output_mean = torch.from_numpy(train_output_mean).float().to('cuda').type(torch.cuda.FloatTensor)
        print('-'*60)
        print('mean')
        print(self.train_output_mean)

        train_output_std = self.data_loader.train_std[-self.xy_split:]
        self.train_output_std = torch.from_numpy(train_output_std).float().to('cuda').type(torch.cuda.FloatTensor)

        print('std')
        print(self.train_output_std)
        print('-'*60)


        self.model = Net(self.data_loader.obs_size, self.data_loader.frc_size, hidden_dim).to('cuda')
        # self.model = BinaryGRFNet(self.data_loader.obs_size, self.data_loader.frc_size, hidden_dim).to('cuda')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        self.criterion = nn.MSELoss()

    def train(self):
        min_val_loss = np.inf

        for epoch in range(self.num_epochs):
            train_loss = 0
            val_loss = 0

            train_data, val_data = self.data_loader.shuffle_split(self.batch_size)

            temp = torch.zeros(0).to('cuda')

            self.model.train()
            for batch in train_data:
                x_trn = batch[:, :-self.xy_split]
                y_trn = batch[:, -self.xy_split:]

                x_trn, y_trn = torch.from_numpy(x_trn).float().to('cuda').type(torch.cuda.FloatTensor), torch.from_numpy(y_trn).float().to('cuda').type(torch.cuda.FloatTensor)

                self.optimizer.zero_grad()
                pred = self.model(x_trn)
                loss = self.criterion(pred, y_trn)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            self.model.eval()
            for batch in val_data:
                x_val = batch[:, :-self.xy_split]
                y_val = batch[:, -self.xy_split:]

                x_val, y_val = torch.from_numpy(x_val).float().to('cuda').type(torch.cuda.FloatTensor), torch.from_numpy(y_val).float().to('cuda').type(torch.cuda.FloatTensor)

                with torch.no_grad():
                    pred = self.model(x_val)

                loss = self.criterion(pred, y_val)
                val_loss += loss.item()
                un_norm_pred = pred * self.train_output_std + self.train_output_mean
                un_norm_yVal = y_val * self.train_output_std + self.train_output_mean

                temp = torch.cat((temp, torch.abs(un_norm_pred-un_norm_yVal)))  # delete later

            train_loss = train_loss / len(train_data)
            val_loss = val_loss / len(val_data)

            # WRONG
            force_residuals = temp[:, 3:6]
            torque_residuals = temp[:, 0:3]

            mean_force_residuals = torch.mean(force_residuals)
            std_force_residuals = torch.std(force_residuals)

            mean_torque_residuals = torch.mean(torque_residuals)
            std_torque_residuals = torch.std(torque_residuals)


            if val_loss < min_val_loss:
                min_val_loss = val_loss
                # torch.save(self.model.state_dict(), 'grfc_model.pt')

            if epoch % 1 == 0:
                print('epoch', epoch, '\t Training Loss: %.4f' % train_loss, ' \t Validation Loss: %.4f' % val_loss, '\t Force Residuals: %.3f' % mean_force_residuals, '+- %.3f' % std_force_residuals, '\t Torque Residuals: %.3f' % mean_torque_residuals, '+- %.3f' % std_torque_residuals,)
