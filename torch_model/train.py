from torch_model.estimator import NewsClassificator
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


def train_classifier(dl_train, dl_valid, num_classes, epochs=10, checkpoint=True):

    model_checkpoint = ModelCheckpoint(dirpath="./runs/pl_classifier",
                                       filename="{epoch}-{val_loss:.3f}",
                                       monitor="val_loss",
                                       mode="min",
                                       save_top_k=1)

    early_stopping = EarlyStopping(monitor="val_loss",
                                   mode="min",
                                   patience=4,
                                   verbose=True,
                                   min_delta=0.01
                                   )

    estimator = NewsClassificator(num_classes=num_classes)

    model = Trainer(
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=checkpoint,
        callbacks=[model_checkpoint, early_stopping],
    )

    model.fit(estimator, dl_train, dl_valid)

    return model
