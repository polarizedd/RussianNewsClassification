from torch.utils.data import Dataset


class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index, :]

        text = row[:-1]
        target = row[-1]

        return text, target
