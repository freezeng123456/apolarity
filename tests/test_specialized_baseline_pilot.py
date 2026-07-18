import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "specialized_pilot", ROOT / "scripts" / "run_specialized_baseline_pilot.py"
)
pilot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pilot)


def test_direct_laplacian_quadratic():
    x = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    y = x.square().sum(dim=1)
    lap = pilot.direct_laplacian(y, x)
    torch.testing.assert_close(lap, torch.full_like(lap, 4.0))


def test_mixed_and_wire_shapes_and_dtypes():
    x = torch.randn(5, 2, dtype=torch.float64)
    mixed = pilot.TanhMLP(2, 16, 2, 2)
    assert mixed(x).shape == (5, 2)

    wire = pilot.WirePINN(2, 16, 2, omega0=10.0, sigma0=10.0)
    out = wire(x)
    assert out.shape == (5, 1)
    assert out.dtype == torch.float64


def test_plane_wave_analytic_laplacian_matches_autograd():
    torch.manual_seed(0)
    model = pilot.PlaneWaveNet(2, 8, init_wavenumber=3.0)
    x = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)
    pred, analytic = model.pred_and_laplacian(x)
    direct_real = pilot.direct_laplacian(pred.real, x)
    direct_imag = pilot.direct_laplacian(pred.imag, x)
    direct = direct_real + 1j * direct_imag
    torch.testing.assert_close(analytic, direct, rtol=1e-10, atol=1e-10)
