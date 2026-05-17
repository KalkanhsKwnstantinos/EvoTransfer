from src.methods import dispatch, dispatch_list, filter_tool

class GeneSpace:

    def __init__(self, config: dict):
        self.config = config

    def __call__(self, key):
        value = self.config[key]
        if 'size' in key and isinstance(value, tuple):
            return filter_tool(*value)
        if value:
            return dispatch(value)
        else:
            if 'conv' in key:
                return dispatch_list(layers=self('conv_layers'), filters=self('conv_size'))
            else:
                return dispatch_list(layers=self('dense_layers'), filters=self('dense_size'))


#ind[] = [lr, batch, dropout, kernel, conv_filters, dense_units]

def setup_gs(lr=0.0005, batch_size=256, dropout_rate=0.5, kernel=3, conv_layers=1, conv_size=128, dense_layers=1, dense_size=128):

    args = locals()
    for v in args:
        if isinstance(v, list | tuple | int | float):
            raise TypeError(f"Variables must be int, float, tuple or list. Error found in variable:{v}")
        
    
    CONFIG = {
        "lr": lr,
        "batch_size": batch_size,
        "dropout_rate": dropout_rate,
        "kernel": kernel,
        "conv_layers": conv_layers,
        "conv_size" : conv_size,
        "dense_layers": dense_layers,
        "dense_size" : dense_size,
        "conv_filters": None,
        "dense_filters": None
    }

    return GeneSpace(CONFIG)
