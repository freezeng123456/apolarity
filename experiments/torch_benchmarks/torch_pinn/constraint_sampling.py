"""Independent uniform IC/BC streams; counts and trace definitions are unchanged."""
import hashlib
import math
import numpy as np


class ConstraintSampler:
    def __init__(self, case, side, seed):
        if case not in ('kdv1d', 'kdv2d', 'ch2d') or side < 2:
            raise ValueError('invalid constraint sampler')
        self.case, self.side = case, side
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 0x49434243]))

    def sample_numpy(self):
        count = self.side if self.case == 'kdv1d' else self.side**2
        dim = 2 if self.case == 'kdv1d' else 3
        lo, hi = (0., math.pi) if self.case == 'ch2d' else (-1., 1.)
        surfaces = [('initial', dim-1, 0.)]
        if self.case == 'kdv1d':
            surfaces += [('left', 0, lo), ('right', 0, hi)]
        elif self.case == 'kdv2d':
            surfaces += [('x_left', 0, lo), ('x_right', 0, hi), ('y_bottom', 1, lo)]
        else:
            surfaces += [('left', 0, lo), ('right', 0, hi), ('bottom', 1, lo), ('top', 1, hi)]
        out = {}
        for name, axis, value in surfaces:
            p = self.rng.random((count, dim)).astype(np.float32)
            p[:, :-1] = lo + (hi-lo)*p[:, :-1]
            p[:, axis] = value
            out[name] = p
        return out

    @property
    def state(self):
        return self.rng.bit_generator.state


def batch_hash(arrays):
    h = hashlib.sha256()
    for name, p in arrays.items():
        h.update(name.encode()); h.update(str(p.shape).encode()); h.update(p.tobytes())
    return h.hexdigest()


def as_tensors(case, arrays, device):
    import torch
    out = {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}
    return tuple(out[k] for k in ('initial', 'left', 'right')) if case == 'kdv1d' else out
