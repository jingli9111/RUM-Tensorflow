def get_config(model):
    if model == 'ptb_fs_rum':
        return ptb_fs_rum_config()
    elif model == 'ptb_rum_single':
        return ptb_rum_single_config()
    elif model == 'ptb_rum_double':
        return ptb_rum_double_config()
    else:
        raise ValueError("Invalid model: %s", model)


class ptb_fs_rum_config(object):
    """PTB config."""
    cell = "fs-rum"
    init_scale = 0.01
    learning_rate = 0.002
    max_grad_norm = 1.0
    num_layers = 2
    num_steps = 150
    cell_size = 700
    hyper_size = 1000
    embed_size = 128
    max_epoch = 200
    max_max_epoch = max_epoch
    keep_prob = 0.65
    zoneout_h = 0.9
    zoneout_c = 0.5
    lr_decay = 0.1
    batch_size = 128
    vocab_size = 50
    fast_layers = 2
    T_norm = 1.0
    use_zoneout = True
    use_layer_norm = True 
    dataset = 'ptb'

class ptb_rum_single_config(object):
    """PTB config."""
    cell = "rum"
    num_steps = 150
    learning_rate = 0.002
    T_norm = 1.0

    num_layers = 1
    init_scale = 0.01
    max_grad_norm = 1.0
    cell_size = 2000
    embed_size = 128
    max_epoch = 100
    max_max_epoch = max_epoch
    keep_prob = 0.65
    zoneout_h = 0.9
    lr_decay = 0.1
    batch_size = 128
    vocab_size = 50
    use_layer_norm = True
    use_zoneout = True
    dataset = 'ptb'

class ptb_rum_double_config(object):
    """PTB config."""
    cell = "rum"
    num_steps = 150
    learning_rate = 0.002
    T_norm = 0.3

    num_layers = 2
    init_scale = 0.01
    max_grad_norm = 1.0
    cell_size = 1500
    embed_size = 128
    max_epoch = 100
    max_max_epoch = max_epoch
    keep_prob = 0.65
    zoneout_h = 0.9
    lr_decay = 0.1
    batch_size = 128
    vocab_size = 50
    use_layer_norm = True
    use_zoneout = True
    dataset = 'ptb'