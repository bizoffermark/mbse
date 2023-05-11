import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/honam/workspace/ode/pyur5/include/mbse')

from mbse.utils.models import ProbabilisticEnsembleModel, FSVGDEnsemble, KDEfWGDEnsemble
import seaborn as sns
sns.reset_defaults()
sns.set_context(context='talk', font_scale=1.0)
from jax.config import config
import wandb 
# config.update('jax_disable_jit', True)
import pickle
import jax.numpy as jnp
import copy
import cloudpickle

def dataset(x, y, batch_size):
    ids = np.arange(len(x))
    while True:
        ids = np.random.choice(ids, batch_size, False)
        yield x[ids].astype(np.float32), y[ids].astype(np.float32)


def plot(x, y, x_tst, y_true, yhats_mean, yhats_std, alpha, name):
    plt.figure(figsize=[15, 4.0], dpi=100)  # inches
    plt.plot(x, y, 'b.', label='observed')
    plt.plot(x_tst, y_true, label='true function', linewidth=1.)
    #for i, yhat_mean in enumerate(yhats_mean):
    #    m = np.squeeze(yhat_mean)
    #    s = np.squeeze(yhats_std[i])
    #    if i < 15:
    m = np.mean(yhats_mean, axis=0)
    eps_s = np.std(yhats_mean, axis=0) * alpha
    eps_al = np.mean(yhats_std, axis=0)
    total_var = np.square(eps_s) + np.square(eps_al)
    total_std = np.sqrt(total_var)
    plt.plot(x_tst.squeeze(), m, 'r', label='ensemble means', linewidth=1.)
    plt.fill_between(x_tst.squeeze(), m - 3 * eps_s, m + 3 * eps_s, color='b', linewidth=0.5, label='3 * epistemic ensemble stdev', alpha=0.4)
    plt.fill_between(x_tst.squeeze(), m - 3 * total_std, m + 3 * total_std, color='g', linewidth=0.5, label='3 * total ensemble stdev', alpha=0.2)
    #    avgm += m
    #plt.plot(x_tst, avgm / len(yhats_mean), 'r', label='overall mean', linewidth=4)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='center left', fancybox=True, framealpha=0., bbox_to_anchor=(0.95, 0.5))
    plt.tight_layout()
    plt.ylim(-3, 3)
    plt.savefig(name, dpi=300)


def normalize(x, mu_x, std_x, eps=1e-8):
    return (x - mu_x)/(std_x + eps)

def denormalize(x, mu_x, std_x, eps=1e-8):
    return (std_x + eps)*x + mu_x

def load_dataset(data, seed=43):
    np.random.seed(seed)

    x = data['train']['x']
    y = data['train']['y']
    x_val = data['valid']['x']
    y_val = data['valid']['y']
    x_tst = data['test']['x']
    y_true = data['test']['y']

    mu_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    mu_y = np.mean(y, axis=0)
    std_y = np.std(y, axis=0)

    x = normalize(x, mu_x, std_x)
    y = normalize(y, mu_y, std_y)

    x_val = normalize(x_val, mu_x, std_x)
    y_val = normalize(y_val, mu_y, std_y)

    x_tst = normalize(x_tst, mu_x, std_x)
    y_true = normalize(y_true, mu_y, std_y)

    metadata = dict()
    metadata['mu_x'] = mu_x
    metadata['std_x'] = std_x
    metadata['mu_y'] = mu_y
    metadata['std_y'] = std_y
    metadata['min_x'] = x.min(axis=0)
    metadata['max_x'] = x.max(axis=0)

    with open("/home/honam/workspace/ode/pyur5/metadata/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    return x.astype(np.float32), y.astype(np.float32), x_val.astype(np.float32), y_val.astype(np.float32), x_tst.astype(np.float32), y_true.astype(np.float32)


with open("/home/honam/workspace/ode/pyur5/data_pkl/data_{}.pkl".format(1), "rb") as f:
    data = pickle.load(f)

# with open("/home/honam/workspace/ode/pyur5/metadata/metadata.pkl", "rb") as f:
#     metadata = pickle.load(f)
batch_size = 256
# x_range = [-20, 60]
# x_range_train = [-10, 30]
x, y, x_val, y_val, x_tst, y_true = load_dataset(data)

print("x.shape", x.shape)


data_train = iter(dataset(x, y, batch_size))

num_train_steps = 20000
ModelName = "ProbabilisticEnsemble"


# name_init = NAME + 'init.png'
# name_end = NAME + 'trained.png'


# predictions = model.predict(x_tst)
# alpha, score = model.calculate_calibration_alpha(params=model.particles, xs=x_val, ys=y_val)
# print(alpha, score)
# yhats_ensemble_mean, yhats_ensemble_std = predictions[..., 0], predictions[..., 1]
# plot(x, y, x_tst, y_true, yhats_ensemble_mean, yhats_ensemble_std, alpha, name=name_init)

num_models = 1
NUM_ENSEMBLES = 5
n_horizon = 50
def train():
    wandb.init(project="ur5_model_train")
    config = wandb.config
    print(config)
    lr = config.lr
    n_layers = config.n_layers
    featrues = config.features * n_layers
    num_ensemble = config.num_ensemble

    if ModelName == "ProbabilisticEnsemble":
        model = ProbabilisticEnsembleModel(
            example_input=x[:batch_size],
            features=featrues,#[32],#[256, 256, 256, 256],
            num_ensemble=num_ensemble,#NUM_ENSEMBLES, 
            lr=lr,#0.0005,
            deterministic=True,
            output_dim=4,
        )
        NAME = 'probabilistic_ensemble_'
    elif ModelName == "fSVGD":
        model = FSVGDEnsemble(
            example_input=x[:batch_size],
            features=[64, 64],
            num_ensemble=NUM_ENSEMBLES,
            lr=0.005,
        )
        NAME = 'fsvgd_ensemble_'
    else:
        model = KDEfWGDEnsemble(
            example_input=x[:batch_size],
            features=[64, 64],
            num_ensemble=NUM_ENSEMBLES,
            lr=0.005,
            #prior_bandwidth=100,
        )
        NAME = 'kde_ensemble_'

    val_loss_best = 1e10
    for i in range(num_train_steps):

        train_loss, train_loss_grad = model.train_step(*next(data_train))

        def loss_fn(x, y):
            y_pred = model.predict(x)
            mu, sig = jnp.split(y_pred, 2, axis=-1)
            loss = jnp.mean((mu.mean(0) - y)**2)
            return loss 
        
        val_loss = loss_fn(x_val, y_val)
        tst_loss = loss_fn(x_tst, y_true)
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            tst_loss_best = tst_loss
            model_best = copy.deepcopy(model)
            with open("/home/honam/workspace/ode/pyur5/model/best_model_{}.pkl".format(wandb.run.name), "wb") as f:
                cloudpickle.dump(model_best, f)
            count = 0
        else:
            count += 1
        if count > 100:
            break
        log = {"train_loss": train_loss, "train_loss_grad": train_loss_grad, "val_loss": val_loss, "tst_loss": tst_loss, "val_loss_best": val_loss_best, "tst_loss_best": tst_loss_best, "epoch": i+1}
        wandb.log(log)
        # print("iter : %2d, train_loss : %5.4f, train_loss_grad: %5.4f, val_loss: %5.4f" % (i, train_loss, train_loss_grad, val_loss))
    def predict(x):
        y_pred = model_best.predict(x)
        mu, sig = jnp.split(y_pred, 2, axis=-1)
        return mu.mean(0)
    # for i in range(n_horizon):
        
    wandb.finish()
# predictions = model.predict(x_tst)
# alpha, score = model.calculate_calibration_alpha(params=model.particles, xs=x_val, ys=y_val)
# yhats_ensemble_mean, yhats_ensemble_std = predictions[..., 0], predictions[..., 1]
# print(alpha, score)
# plot(x, y, x_tst, y_true, yhats_ensemble_mean, yhats_ensemble_std, alpha, name=name_end)

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val_loss_best',
        'goal': 'minimize'
    },
    'parameters': {
        'num_ensemble': {
            'values': [10]#[5, 10, 20]
        },
        'lr': {
            'values': [0.0005]#, 0.001, 0.005]
        },
        'features': {
            'values': [[32]]#, [64], [128], [256]]
        },
        'n_layers': {
            'values': [3]#[1, 2, 3, 4]
        },
    }
}

sweep_id = wandb.sweep(sweep_config, project="ur5_model_train")
wandb.agent(sweep_id, function=train)
