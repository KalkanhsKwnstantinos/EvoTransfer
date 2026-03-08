from src.model import build_model
import tensorflow as tf

def evaluate_individual(individual, dataset):

    hparams = individual_to_dict(individual)

    (x_train, y_train), (x_val, y_val) = dataset

    model = build_model(
        input_shape=x_train.shape[1:],
        num_classes=10,
        hparams=hparams
    )

    model.fit(
        x_train, y_train,
        epochs=3,        # keep small for GA
        batch_size=hparams["batch_size"],
        verbose=0
    )

    loss, acc = model.evaluate(x_val, y_val, verbose=0)

    return (acc,)   # DEAP expects tuple


def individual_to_dict(ind):
    return {
        "lr": ind[0],
        "batch_size": int(ind[1]),
        "dropout": ind[2],
        "filters": int(ind[3]),
        "dense_units": int(ind[4]),
        "optimizer": "adam" if ind[5] < 0.5 else "sgd",
        "momentum": ind[6]
    }