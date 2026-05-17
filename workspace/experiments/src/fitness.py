from src.model import build_model
import tensorflow as tf
import gc, time
early_stopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',        # tensor already uses val_loss for modeling 
    patience=3,                # epochs to wait for betterment
    restore_best_weights=True  # Reverting to best model, not last
)

def evaluate_individual(ind, dataset):
    #ind[] = [lr, batch, dropout, kernel, conv_filters, dense_units]
    (train_ds, val_ds), (shape, num_classes) = dataset
    
    train_ds = train_ds.batch(ind[1]).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(ind[1]).prefetch(tf.data.AUTOTUNE)
    
    model = build_model(ind=ind, shape=shape, num_classes=num_classes)

    start = time.time()
    
    model.fit(
        train_ds,
        epochs=100, #set high since early_stopper is part of the architecture
        validation_data=val_ds,
        callbacks=[early_stopper],
        verbose=0
    )

    elapsed = time.time() - start
    
    loss, acc = model.evaluate(val_ds, verbose=0)

    #clean-up after evaluation
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    
    return (acc, loss, elapsed)   # Tuple, acc max and elapsed min
