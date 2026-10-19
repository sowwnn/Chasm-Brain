"""Training loop (spec §6): sample (t_prev, t) per batch, teacher-force c_prev = project(v_true, t_prev) + N(0, 0.05),
penalise the new band. AdamW lr 3e-4, batch 64, early stopping on val (patience 15, max 150 epochs).
Writes config.yaml + metrics.json per run; preds.npy on test written once at the end."""


def sample_t_pair(schedule_kind: str, n_steps: int, continuous: bool = False):
    """From the fixed schedule, or t_prev ~ U[0,1), t ~ U(t_prev, 1] if continuous."""
    raise NotImplementedError


def make_batch(batch, basis, t_prev, t, noise_sigma: float = 0.05):
    raise NotImplementedError


def scheduled_sampling(p: float):
    """After epoch 50, replace true c_prev with the model's own with prob p, p: 0 -> 0.5. Mandatory ablation."""
    raise NotImplementedError


def train_epoch(model, loader, optim, cfg):
    raise NotImplementedError


def validate(model, loader, cfg):
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
