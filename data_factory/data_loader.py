import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler


class BaseSegLoader(Dataset):
    def __init__(self, win_size, step, mode="train"):
        super().__init__()
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        self.train = None
        self.val = None
        self.test = None
        self.test_labels = None

    def __len__(self):
        if self.mode == "train":
            data = self.train
            step = self.step
        elif self.mode == "val":
            data = self.val
            step = self.step
        elif self.mode == "test":
            data = self.test
            step = self.step
        else:  # thre
            data = self.test
            step = self.win_size

        return (data.shape[0] - self.win_size) // step + 1

    def __getitem__(self, index):
        if self.mode == "train":
            start = index * self.step
            end = start + self.win_size
            x = self.train[start:end]
            y = self.test_labels[0:self.win_size]
            return np.float32(x), np.float32(y)

        elif self.mode == "val":
            start = index * self.step
            end = start + self.win_size
            x = self.val[start:end]
            y = self.test_labels[0:self.win_size]
            return np.float32(x), np.float32(y)

        elif self.mode == "test":
            start = index * self.step
            end = start + self.win_size
            x = self.test[start:end]
            y = self.test_labels[start:end]
            return np.float32(x), np.float32(y)

        else:  # thre
            start = index * self.win_size
            end = start + self.win_size
            x = self.test[start:end]
            y = self.test_labels[start:end]
            return np.float32(x), np.float32(y)


class PSMSegLoader(BaseSegLoader):
    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(win_size, step, mode)

        train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
        train_data = train_df.values[:, 1:]
        train_data = np.nan_to_num(train_data)

        self.scaler.fit(train_data)
        self.train = self.scaler.transform(train_data)

        test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
        test_data = test_df.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)

        self.val = self.test
        self.test_labels = pd.read_csv(
            os.path.join(data_path, "test_label.csv")
        ).values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)


class MSLSegLoader(BaseSegLoader):
    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(win_size, step, mode)

        train_data = np.load(os.path.join(data_path, "MSL_train.npy"))
        self.scaler.fit(train_data)
        self.train = self.scaler.transform(train_data)

        test_data = np.load(os.path.join(data_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)

        self.val = self.test
        self.test_labels = np.load(os.path.join(data_path, "MSL_test_label.npy"))

        print("test:", self.test.shape)
        print("train:", self.train.shape)


class SMAPSegLoader(BaseSegLoader):
    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(win_size, step, mode)

        train_data = np.load(os.path.join(data_path, "SMAP_train.npy"))
        self.scaler.fit(train_data)
        self.train = self.scaler.transform(train_data)

        test_data = np.load(os.path.join(data_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)

        self.val = self.test
        self.test_labels = np.load(os.path.join(data_path, "SMAP_test_label.npy"))

        print("test:", self.test.shape)
        print("train:", self.train.shape)


class SMDSegLoader(BaseSegLoader):
    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(win_size, step, mode)

        train_data = np.load(os.path.join(data_path, "SMD_train.npy"))
        self.scaler.fit(train_data)
        self.train = self.scaler.transform(train_data)

        test_data = np.load(os.path.join(data_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        self.test_labels = np.load(os.path.join(data_path, "SMD_test_label.npy"))

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("val:", self.val.shape)


class GenericNPYSegLoader(BaseSegLoader):
    """
    Generic loader for datasets stored as:
      x_train.npy
      x_test.npy
      y_test.npy
    """
    def __init__(self, data_path, win_size, step, mode="train", val_source="test"):
        super().__init__(win_size, step, mode)

        train_file = os.path.join(data_path, "x_train.npy")
        test_file = os.path.join(data_path, "x_test.npy")
        label_file = os.path.join(data_path, "y_test.npy")

        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Missing file: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Missing file: {test_file}")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Missing file: {label_file}")

        train_data = np.load(train_file)
        test_data = np.load(test_file)
        test_labels = np.load(label_file)

        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)
        test_labels = np.nan_to_num(test_labels)

        if train_data.ndim != 2:
            raise ValueError(f"x_train.npy must be 2D, got shape {train_data.shape}")
        if test_data.ndim != 2:
            raise ValueError(f"x_test.npy must be 2D, got shape {test_data.shape}")

        self.scaler.fit(train_data)
        self.train = self.scaler.transform(train_data)
        self.test = self.scaler.transform(test_data)

        if val_source == "test":
            self.val = self.test
        elif val_source == "train_tail":
            data_len = len(self.train)
            self.val = self.train[int(data_len * 0.8):]
        else:
            raise ValueError(f"Unsupported val_source: {val_source}")

        self.test_labels = np.asarray(test_labels).reshape(-1, 1)

        if len(self.test_labels) != len(self.test):
            raise ValueError(
                f"Label length mismatch: len(y_test)={len(self.test_labels)} "
                f"but len(x_test)={len(self.test)}"
            )

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("val:", self.val.shape)


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if dataset == 'SMD':
        dataset_obj = SMDSegLoader(data_path, win_size, step, mode)
    elif dataset == 'MSL':
        dataset_obj = MSLSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'SMAP':
        dataset_obj = SMAPSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'PSM':
        dataset_obj = PSMSegLoader(data_path, win_size, 1, mode)
    elif dataset in ['CreditCard', 'CyberSecurity', 'FallingPeople', 'SWaT']:
        dataset_obj = GenericNPYSegLoader(
            data_path=data_path,
            win_size=win_size,
            step=1 if mode in ['val', 'test', 'thre'] else step,
            mode=mode,
            val_source="test"
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    shuffle = (mode == 'train')

    data_loader = DataLoader(
        dataset=dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False
    )

    return data_loader