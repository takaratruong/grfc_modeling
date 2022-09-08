import numpy as np
from pathlib import Path
import pprint

from typing import List
from typing import Dict
from typing import Tuple
from scipy.interpolate import interp1d

class DataLoader:
    def __init__(self, path, args=None) -> None:

        self.percent_train = .9

        # all input features and outputs normalized wrt train data
        self.norm_train_data, self.train_mean, self.train_std, self.norm_val_data = self.load_data(path)

        print('-'*60)
        print('mean')
        print(self.train_mean)
        print(self.train_mean.shape)

        print('std')
        print(self.train_std)
        print('-'*60)

        # print(self.train_data.shape)
        self.obs_size = 69 + 28
        self.frc_size = 12  # CHANGE THIS

    def normalize_input(self, input):
        assert input.shape[1] == self.obs_size, 'Wrong input dim'
        norm_input = (input - self.train_mean[:-12])/self.train_std[:-12]
        return norm_input

    def un_normalize_output(self, norm_output):
        assert norm_output.shape[1] == self.frc_size, 'Wrong input dim'
        output = norm_output * self.train_std[-12:] + self.train_mean[-12:]
        return output

    def load_data(self, path): # -> Tuple[np.ndarray, np.ndarray]:

        # Stack all processed data

        data = np.load(path)
        print(data.shape)
        # data = np.array([])
        # for np_name in Path(path).glob('*.np[yz]'):
        #     traj = np.load(np_name)
        #     data = np.vstack([data, traj]) if data.size else traj
        #
        # assert len(data) > 0, 'No data loaded'        # lf = abs(data[:, 70+3])
        #
        # lf = np.array(lf > 0)
        # lf = lf.astype(float)
        #
        #
        # rf = abs(data[:, 70+6+3])
        # rf = np.array(rf > 0)
        # rf = rf.astype(float)
        #
        # data = data[:, 0:70].copy()
        # data = np.hstack((data, lf.reshape(-1,1), rf.reshape(-1,1)))

        # Split into train and validation sets
        n = int(len(data) * self.percent_train)

        train = data[:n, :]
        val = data[n:, :]

        # Normalize Data
        train_mean = np.mean(train, axis=0)
        train_std = np.std(train, axis=0)

        norm_train = (train - train_mean)/train_std
        norm_val = (val-train_mean)/train_std

        return norm_train, train_mean, train_std, norm_val

    def shuffle_split(self, approx_batch_size):
        d_train = max(1, len(self.norm_train_data) // approx_batch_size)
        d_val = max(1, len(self.norm_val_data) // approx_batch_size)

        np.random.shuffle(self.norm_train_data)

        trn = np.array_split(self.norm_train_data, d_train, axis=0)
        val = np.array_split(self.norm_val_data, d_val, axis=0)

        return trn, val

if __name__ == "__main__":
    path = '/home/takaraet/fc_modeling/grfc_data.npy'

    mocap = DataLoader(path)

    trn, val = mocap.shuffle_split(50)

    # print(val)
