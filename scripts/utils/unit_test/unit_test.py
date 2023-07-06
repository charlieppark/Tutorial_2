import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_nerf.src.signal_encoder.positional_encoder import PositionalEncoder
from torch_nerf.src.renderer.ray_samplers.stratified_sampler import StratifiedSampler
from torch_nerf.src.cameras.rays import RaySamples, RayBundle
from torch_nerf.src.renderer.integrators.quadrature_integrator import QuadratureIntegrator
from test_func import e, s, ry, dd, kk

try:
    embed_fn, input_ch = e(10, True)
    positional_encoder = PositionalEncoder(1, 10, True)
    for i in range(100):
        randin = torch.rand(3)
        a = embed_fn((torch.Tensor([i, i+1, i*2])))
        b = positional_encoder.encode(torch.Tensor([i, i+1, i*2]))
        t = torch.allclose(a, b, atol=1e-10)
        assert t
except AssertionError:
    embed_fn, input_ch = e(10, False)
    positional_encoder = PositionalEncoder(1, 10, True)
    for i in range(100):
        randin = torch.rand(3)
        a = embed_fn((torch.Tensor([i, i+1, i*2])))
        b = positional_encoder.encode(torch.Tensor([i, i+1, i*2]))
        t = torch.allclose(a, b, atol=1e-10)
        assert t, "Error : positional_encoder"

batch_size = 10
origins = torch.rand(batch_size, 3) 
directions = torch.rand(batch_size, 3) 
near = 2.
far = 6.
n_samples = 2
nears = torch.ones((batch_size * batch_size)) * near
fars = torch.ones((batch_size * batch_size)) * far

st = StratifiedSampler()

ray_bundle = RayBundle(origins, directions, nears, fars)

torch.manual_seed(0)
a = st.sample_along_rays(ray_bundle, n_samples).t_samples.to('cpu')
torch.manual_seed(0)
b = s(batch_size, origins, directions, near, far, n_samples)
t = torch.allclose(a, b, atol=1e-10)
assert t, "Error : sample_along_rays_uniform"

t_samples = a
ray_samples = RaySamples(ray_bundle, t_samples)

a = ray_samples.compute_sample_coordinates()
b = ry(origins, directions, t_samples)
t = torch.allclose(a, b, atol=1e-10)
assert t, "Error : compute_sample_coordinates"

ray_samples.t_samples = ray_samples.t_samples.to(torch.cuda.current_device())
a = ray_samples.compute_deltas().to('cpu')
b = dd(t_samples)
t = torch.allclose(a, b, atol=1e-10)
assert t, "Error : compute_deltas"

sigma = torch.rand(batch_size, n_samples)
radiance = torch.rand(batch_size, n_samples, 3)
delta = a

quadrature_integrator = QuadratureIntegrator()
rgbs_a, weights_a = kk(sigma, radiance, delta)
rgbs_b, weights_b = quadrature_integrator.integrate_along_rays(sigma.to(torch.cuda.current_device()), radiance.to(torch.cuda.current_device()), delta.to(torch.cuda.current_device()))

rgbs_t = torch.allclose(rgbs_a, rgbs_b.to('cpu'), atol=1e-10)
weights_t = torch.allclose(weights_a, weights_b.to('cpu'), atol=1e-10)

assert rgbs_t and weights_t, "Error : quadrature_integrator (torch.cumprod)"
