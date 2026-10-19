"""Fixed and dynamic t schedules (spec §5, §7)."""
import numpy as np
import torch


def fixed_schedule(n_steps: int, kind: str = "log") -> list:
    """n_steps=1 -> [1.0] (single-shot baseline). n_steps>1, kind='linear' -> evenly spaced in (0,1].
    kind='log' -> geometrically spaced so early steps add few dimensions, matching the spec example
    (n_steps=4, log -> approx [0.02, 0.07, 0.33, 1.0])."""
    if n_steps == 1:
        return [1.0]
    if kind == "linear":
        return [round((i + 1) / n_steps, 4) for i in range(n_steps)]
    if kind == "log":
        # geometric sequence from a small t0 to 1.0
        t0 = 0.02
        vals = np.geomspace(t0, 1.0, n_steps)
        vals[-1] = 1.0
        return [round(float(v), 4) for v in vals]
    raise ValueError("kind must be 'linear' or 'log'")


def run_fixed(model, cls, patch, schedule):
    """Deterministic inference: c = zeros; for t in schedule: c = model.step(...); return basis.lift(c, 1.0)."""
    B = cls.shape[0]
    c = torch.zeros(B, 0, device=cls.device, dtype=cls.dtype)
    t_prev = 0.0
    for t in schedule:
        c = model.step(cls, patch, c, t_prev, t)
        t_prev = t
    return model.predict_voxels(c, 1.0)


def run_dynamic(model, cls, patch, schedule, halt_head, threshold: float = 0.5):
    """Stop when p_halt > threshold (evaluated after each step using the Refiner's fused hidden state).
    Returns (v_hat, n_steps_used [B]). halt_head must be trained on the frozen Refiner (spec §7, phase 2)."""
    B = cls.shape[0]
    c = torch.zeros(B, 0, device=cls.device, dtype=cls.dtype)
    t_prev = 0.0
    n_steps_used = torch.zeros(B, dtype=torch.long, device=cls.device)
    halted = torch.zeros(B, dtype=torch.bool, device=cls.device)
    v_hat = None
    for step_i, t in enumerate(schedule):
        z = model.compute_z(cls, patch, c, t_prev, t)   # same trunk forward() uses, before appending c_new
        c_new = model.forward(cls, patch, c, t_prev, t)
        c = torch.cat([c, c_new], dim=1)
        p_halt = halt_head(z).squeeze(-1)
        newly_halted = (p_halt > threshold) & ~halted
        n_steps_used[newly_halted] = step_i + 1
        halted = halted | newly_halted
        t_prev = t
        if halted.all():
            break
    n_steps_used[~halted] = len(schedule)
    v_hat = model.predict_voxels(c, 1.0)
    return v_hat, n_steps_used
