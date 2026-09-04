import pytest


def test_lite_unet_forward_cpu():
    torch = pytest.importorskip("torch")
    from src.models.lite.scheduler import DDPMScheduler
    from src.models.lite.unet import LiteCondUNet

    model = LiteCondUNet(base=16)
    sched = DDPMScheduler(timesteps=10)
    x0 = torch.randn(2, 3, 32, 32)
    lq = torch.randn(2, 3, 32, 32)
    t = torch.tensor([3, 7])
    noise = torch.randn_like(x0)
    xt = sched.q_sample(x0, t, noise)
    pred = model(xt, lq, t)
    assert pred.shape == x0.shape
