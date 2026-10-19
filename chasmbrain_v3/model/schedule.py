"""Fixed and dynamic t schedules (spec §5, §7)."""


def fixed_schedule(n_steps: int, kind: str = "log") -> list:
    """e.g. n_steps=4, log -> [0.02, 0.07, 0.33, 1.0]; n_steps=1 -> [1.0] (single-shot baseline)."""
    raise NotImplementedError


def run_fixed(model, cls, patch, schedule):
    """Deterministic inference: c = zeros; for t in schedule: c = model.step(...); return basis.lift(c, 1.0)."""
    raise NotImplementedError


def run_dynamic(model, cls, patch, schedule, threshold: float = 0.5):
    """Stop when p_halt > threshold. Returns (v_hat, n_steps_used)."""
    raise NotImplementedError
