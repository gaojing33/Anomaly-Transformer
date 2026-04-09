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
            return len(self.train)
        elif self.mode == "val":
            return len(self.val)
        elif self.mode == "test":
            return len(self.test)
        else:  # thre
            return len(self.test)

    def __getitem__(self, index):
        raise NotImplementedError


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

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        if self.mode == "train":
            start = index * self.step
            end = start + self.win_size
            return np.float32(self.train[start:end]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == "val":
            start = index * self.step
            end = start + self.win_size
            return np.float32(self.val[start:end]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == "test":
            start = index * self.step
            end = start + self.win_size
            return np.float32(self.test[start:end]), np.float32(self.test_labels[start:end])
        else:
            start = index * self.win_size
            end = start + self.win_size
            return np.float32(self.test[start:end]), np.float32(self.test_labels[start:end])


class MSLSegLoader(PSMSegLoader):
    def __init__(self, data_path, win_size, step, mode="train"):
        BaseSegLoader.__init__(self, win_size, step, mode)
        train_data = np.load(os.path.join(data_path, "MSL_train.npy"))
        self.scaler.fit(train_data)
        self.train = self.scaler.transform(train_data)

        test_data = np.load(os.path.join(data_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)

        self.val = self.test
        self.test_labels = np.load(os.path.join(data_path, "MSL_test_label.npy"))

        print("test:", self.test.shape)
        print("train:", self.train.shape)


class SMAPSegLoader(PSMSegLoader):
    def __init__(self, data_path, win_size, step, mode="train"):
        BaseSegLoader.__init__(self, win_size, step, mode)
        train_data = np.load(os.path.join(data_path, "SMAP_train.npy"))
        self.scaler.fit(train_data)
        self.train = self.scaler.transform(train_data)

        test_data = np.load(os.path.join(data_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)

        self.val = self.test
        self.test_labels = np.load(os.path.join(data_path, "SMAP_test_label.npy"))

        print("test:", self.test.shape)
        print("train:", self.train.shape)


class SMDSegLoader(PSMSegLoader):
    def __init__(self, data_path, win_size, step, mode="train"):
        BaseSegLoader.__init__(self, win_size, step, mode)
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


class PreWindowedNPYSegLoader(BaseSegLoader):
    """
    For datasets already windowed as:
      x_train.npy : (N_train, L, C)
      x_test.npy  : (N_test, L, C)
      y_test.npy  : (N_test,) or (N_test, L)
    """

    def __init__(self, data_path, win_size, step, mode="train", val_ratio=0.2):
        super().__init__(win_size, step, mode)

        x_train = np.load(os.path.join(data_path, "x_train.npy"))
        x_test = np.load(os.path.join(data_path, "x_test.npy"))
        y_test = np.load(os.path.join(data_path, "y_test.npy"))

        x_train = np.nan_to_num(x_train)
        x_test = np.nan_to_num(x_test)
        y_test = np.nan_to_num(y_test)

        if x_train.ndim != 3:
            raise ValueError(f"x_train.npy must be 3D, got shape {x_train.shape}")
        if x_test.ndim != 3:
            raise ValueError(f"x_test.npy must be 3D, got shape {x_test.shape}")

        n_train, L_train, c_train = x_train.shape
        n_test, L_test, c_test = x_test.shape

        if L_train != win_size or L_test != win_size:
            raise ValueError(
                f"win_size={win_size}, but x_train/x_test have window length "
                f"{L_train}/{L_test}"
            )
        if c_train != c_test:
            raise ValueError(f"train/test feature dims mismatch: {c_train} vs {c_test}")

        # fit scaler on all training points across all windows
        self.scaler.fit(x_train.reshape(-1, c_train))

        x_train_scaled = self.scaler.transform(x_train.reshape(-1, c_train)).reshape(n_train, L_train, c_train)
        x_test_scaled = self.scaler.transform(x_test.reshape(-1, c_test)).reshape(n_test, L_test, c_test)

        # split train -> train/val to avoid using test set as validation threshold source
        n_val = max(1, int(n_train * val_ratio))
        n_val = min(n_val, n_train - 1) if n_train > 1 else 1

        self.train = x_train_scaled[:-n_val] if n_train > 1 else x_train_scaled
        self.val = x_train_scaled[-n_val:] if n_train > 1 else x_train_scaled
        self.test = x_test_scaled

        # y_test handling
        # case 1: window-level labels, shape (N_test,)
        if y_test.ndim == 1:
            if len(y_test) != n_test:
                raise ValueError(
                    f"y_test length {len(y_test)} != number of test windows {n_test}"
                )
            self.test_labels = np.repeat(y_test[:, None], win_size, axis=1)

        # case 2: already per-timestep labels, shape (N_test, L)
        elif y_test.ndim == 2:
            if y_test.shape != (n_test, win_size):
                raise ValueError(
                    f"y_test shape {y_test.shape} incompatible with x_test shape {x_test.shape}"
                )
            self.test_labels = y_test

        else:
            raise ValueError(f"Unsupported y_test shape: {y_test.shape}")

        print("train:", self.train.shape)
        print("val:", self.val.shape)
        print("test:", self.test.shape)
        print("test_labels:", self.test_labels.shape)

    def __getitem__(self, index):
        if self.mode == "train":
            x = self.train[index]
            y = np.zeros((self.win_size,), dtype=np.float32)
            return np.float32(x), np.float32(y)

        elif self.mode == "val":
            x = self.val[index]
            y = np.zeros((self.win_size,), dtype=np.float32)
            return np.float32(x), np.float32(y)

        elif self.mode == "test":
            x = self.test[index]
            y = self.test_labels[index]
            return np.float32(x), np.float32(y)

        else:  # thre
            x = self.test[index]
            y = self.test_labels[index]
            return np.float32(x), np.float32(y)


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
        dataset_obj = PreWindowedNPYSegLoader(
            data_path=data_path,
            win_size=win_size,
            step=step,
            mode=mode,
            val_ratio=0.2
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