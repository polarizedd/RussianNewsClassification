from torch_model.train import NewsClassificator
import torch
import numpy as np
from sklearn.metrics import f1_score


def make_prediction(dl_test, path_to_model):

    prediction = []
    gt = []

    model = NewsClassificator.load_from_checkpoint(path_to_model, num_classes=8)
    model.eval()

    with torch.no_grad():
        for batch in dl_test:
            x, y = batch
            preds = model(x).argmax(dim=1)
            prediction.append(preds)
            gt.append(y)

    prediction = np.concatenate(prediction)
    gt = np.concatenate(gt)

    print('F1 macro score on test = ', f1_score(gt, prediction, average='macro'))

    return prediction
