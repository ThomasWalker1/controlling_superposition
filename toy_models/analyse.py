from utils import *
import os

DEVICE="cpu"

NEGATIVE_SLOPES=np.linspace(0,1,5)

def generate_batch(n_batch, n_features, feat_probability):
    feat = torch.rand((n_batch, n_features))
    batch = torch.where(
        torch.rand((n_batch, n_features)) <= feat_probability,
        feat,
        torch.zeros((),),
    )
    return batch


for negative_slope in NEGATIVE_SLOPES:

    model_name=str(round(negative_slope,ndigits=4)).replace(".","")

    config = Config(
        c_id = 1,
        n_features = 5,
        n_hidden = 2,
        n_instances = 2,
        negative_slope = negative_slope,
        feature_probability = 1/20
    )

    save_config(config)

    model = Model(
        config=config,
        device=DEVICE,
        importance = (0.9**torch.arange(config.n_features))
    )

    if not(os.path.exists(f"superposition/toy_models/state_dicts_{config.c_id}")):
        os.makedirs(f"superposition/toy_models/state_dicts_{config.c_id}")
    if not(os.path.exists(f"superposition/toy_models/importance_{config.c_id}")):
        os.makedirs(f"superposition/toy_models/importance_{config.c_id}")
    if not(os.path.exists(f"superposition/toy_models/activations_{config.c_id}")):
        os.makedirs(f"superposition/toy_models/activations_{config.c_id}")

    optimize(model)

    torch.save(model.state_dict(),f"superposition/toy_models/state_dicts_{config.c_id}/{model_name}.pt")
    torch.save(model.importance,f"superposition/toy_models/importance_{config.c_id}/{model_name}.pt")

    batch=generate_batch(64,config.n_features,config.feature_probability)

    torch.save(model(batch)[1],f"superposition/toy_models/activations_{config.c_id}/{model_name}.pt")

plot_instances(1)