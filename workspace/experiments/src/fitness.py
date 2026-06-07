import multiprocessing as mp

def eval_worker(ind, dataset, queue):

    from src.model import build_model
    import tensorflow as tf
    import gc, time

    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',        # tensor already uses val_loss for modeling 
        patience=3,                # epochs to wait for betterment
        restore_best_weights=True  # Reverting to best model, not last
    )

    model = None
    train_ds = val_ds = None
    
    #ind[] = [lr, batch, dropout, kernel, conv_filters, dense_units]
    try:
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
        queue.put((acc, loss, elapsed))
        
    finally:
        #clean-up after evaluation
        del model, train_ds, val_ds
        tf.keras.backend.clear_session()
        gc.collect()

def evaluate_individual(ind, dataset):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(target=eval_worker, args=(ind, dataset, queue))
    p.start()
    p.join()
    return queue.get()