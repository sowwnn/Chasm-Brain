"""Same basis, one step t = 1: Refiner with n_steps = 1 and identical parameters (spec §9)."""
from ..model.encoder import DualPathEncoder
from ..model.refiner import Refiner
from ..model.schedule import fixed_schedule


def build(basis, d_in: int, d_hidden: int = 256, t_embed_dim: int = 32, n_patches: int = 256):
    """A Refiner trained with the single-step schedule [1.0] -- same architecture and parameter count
    as the multi-step model at the same d_hidden, differing only in how many steps it is trained/run for."""
    encoder = DualPathEncoder(d_in, d_hidden, n_patches=n_patches)
    model = Refiner(encoder, basis, d_hidden=d_hidden, t_embed_dim=t_embed_dim)
    return model, fixed_schedule(1)


if __name__ == "__main__":
    raise NotImplementedError
