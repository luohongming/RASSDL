import torch.utils.data as data
import h5py
import torch


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()

        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.original = f['original'][:, :, :, :]
            self.equlized = f['equlized'][:, :, :, :]

    def __getitem__(self, index):
        input_ = self.original[index, :, :, :]
        target_ = self.equlized[index, :, :, :]
        input_ = torch.from_numpy(input_).float()
        target_ = torch.from_numpy(target_).float()
        return {'input': input_, 'target': target_}

    def __len__(self):
        return len(self.original)