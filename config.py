import model

mnist = {
    'verbose': True,
    'model': model.get_model,
    's_opt': {
        'learning_rate': 0.03,
        'momentum': 0.9,
        'nesterov': True
    },
    't_opt': {
        'learning_rate': 0.03,
        'momentum': 0.9,
        'nesterov': True
    },
    'uda_args': {
        'augmentation_methods': [
            ('brightness',  {'max_delta': 0.2}),
            ('hue',         {'max_delta': 0.2})
        ],
        'out_shape': (28,28,3),
    },
    'data_args': {
        'classes': [str(i) for i in range(10)],
        'out_shape': (28,28,3),
        'batch_size': 512,
        'buffer_size': 512,
        'unlabel_batch_size': 4096
    }
}