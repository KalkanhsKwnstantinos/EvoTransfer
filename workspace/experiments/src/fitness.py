from src.model import build_model
import tensorflow as tf

def evaluate_individual(individual, dataset):

    hparams = individual_to_dict(individual)
    
    (x_train, y_train), (x_val, y_val) = dataset

    model = build_model(hparams['filters'], hparams['lr'])

    model.fit(
        x_train, y_train,
        epochs=3,        # keep small for GA
        batch_size=hparams["batch"],
        verbose=0
    )

    loss, acc = model.evaluate(x_val, y_val, verbose=0)

    #clean-up after evaluation
    del model
    tf.keras.backend.clear_session()
    
    return (acc,)   # DEAP expects tuple


def individual_to_dict(ind):
    return {
        "lr": float(ind[0]),
        "batch": int(ind[1]),
        "filters": list(ind[2])
    }