

<<<<<<< HEAD
def build_model(cf, input_shape):
=======
def build_model(cf):
>>>>>>> b1250baaa20a9bb578d8d052b6ec67bd5aa80232
    conditional_log_probs_model = None
    if cf.model_name == "rbm":
        from ..architectures.rbm import rbm
        model = rbm(cf)
    elif cf.model_name == "drbm":
        from ..architectures.drbm import drbm
        model = drbm(cf)
<<<<<<< HEAD
    elif cf.model_name == "ar1":
        from ..architectures.ar1 import ar
        model, conditional_log_probs_model = ar(cf, input_shape)
    elif cf.model_name == "ar2":
        from ..architectures.ar2 import ar
        model, conditional_log_probs_model = ar(cf, input_shape)
=======
    elif cf.model_name == "ar":
        from ..architectures.ar import ar
        model, conditional_log_probs_model = ar(cf)
>>>>>>> b1250baaa20a9bb578d8d052b6ec67bd5aa80232
    elif cf.model_name == "my_rbm":
        from ..architectures.my_rbm import my_rbm
        model = my_rbm(cf)
    model.summary()
    return model, conditional_log_probs_model


def load_model(cf, model, loadpath):
    if cf.num_gpu<2:
        bad_state_dict = torch.load(loadpath,map_location='cpu')
        correct_state_dict = {re.sub(r'^module\.', '', k): v for k, v in
                                bad_state_dict.items()}
        model.load_state_dict(correct_state_dict)
    else:
        model.load_state_dict(torch.load(loadpath))
    model.eval()
    model.zero_grad()
    return model

