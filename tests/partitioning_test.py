# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.partition_spec import UnreducedKind
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P, auto_axes
import numpy as np

config.parse_flags_with_absl()

jtu.request_cpu_devices(8)


@jtu.pytest_mark_if_available('multiaccelerator')
class ShardMapPartitioningTest(jtu.JaxTestCase):

  @parameterized.product(
      op=((jnp.max, np.max), (jnp.min, np.min), (jnp.sum, np.sum)),
      case=(
          (None, P('x', 'y'), P(), True),
          (0, P('x', 'y'), P('y'), True),
          (0, P(('x', 'y'), None), P(None), True),
          (1, P(('x', 'y'), None), P(('x', 'y')), False),
      ),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reduce(self, op, case, mesh):
    jnp_op, np_op = op
    axis, in_spec, out_spec, has_collective = case
    np_inp = np.arange(32, dtype=np.float32).reshape(4, 8)
    arr = jax.device_put(np_inp, in_spec)

    def f(x):
      self.assertEqual(jax.typeof(x).sharding.spec, in_spec)
      y = jnp_op(x, axis=axis)
      self.assertEqual(jax.typeof(y).sharding.spec, out_spec)
      return y

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_op(np_inp, axis=axis))

    f_jit = jax.jit(f)
    with config.shard_map_partitioning(False):
      self.assertNotIn('sdy.manual_computation', f_jit.lower(arr).as_text())
    self.assertEqual(
        'sdy.manual_computation' in f_jit.lower(arr).as_text(), has_collective)

    out = f_jit(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np_op(np_inp, axis=axis))

    grad = jax.jit(jax.grad(lambda x: jnp.sum(f(x))))(arr)
    self.assertEqual(grad.sharding, arr.sharding)
    self.assertAllClose(
        grad, jax.grad(lambda x: jnp.sum(jnp_op(x, axis=axis)))(np_inp))

  @parameterized.named_parameters(
      ('max', jnp.max, np.max),
      ('min', jnp.min, np.min),
      ('sum', jnp.sum, np.sum),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh(
      (2, 2), ('x', 'y'), axis_types=(AxisType.Explicit, AxisType.Auto))
  def test_auto_mesh_mix(self, jnp_op, np_op, mesh):
    np_inp = np.arange(32, dtype=np.float32).reshape(4, 8)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x', 'y')))

    @jax.jit
    def f(x):
      self.assertEqual(jax.typeof(x).sharding.spec, P('x', None))
      y0 = jnp_op(x, axis=0)
      self.assertEqual(jax.typeof(y0).sharding.spec, P(None))
      y1 = jnp_op(x, axis=1)
      self.assertEqual(jax.typeof(y1).sharding.spec, P('x'))
      y01 = jnp_op(x, axis=(0, 1))
      self.assertEqual(jax.typeof(y01).sharding.spec, P())
      return y0, y1, y01

    y0, y1, y01 = f(arr)
    self.assertArraysEqual(y0, np_op(np_inp, axis=0))
    self.assertEqual(y1.sharding.spec, P('x'))
    self.assertArraysEqual(y1, np_op(np_inp, axis=1))
    self.assertArraysEqual(y01, np_op(np_inp))

  @parameterized.named_parameters(
      ('max', jnp.max, np.max),
      ('min', jnp.min, np.min),
      ('sum', jnp.sum, np.sum),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_auto_axes_decorator(self, jnp_op, np_op, mesh):
    np_inp = np.arange(32, dtype=np.float32).reshape(4, 8)
    arr = jax.device_put(np_inp, P('x', 'y'))

    @partial(auto_axes, axes='x', out_sharding=P())
    def h(a):
      self.assertEqual(jax.typeof(a).sharding.spec, P(None, 'y'))
      out = jnp_op(a, axis=(0, 1))
      self.assertEqual(jax.typeof(out).sharding.spec, P())
      return out

    @jax.jit
    def f(x):
      out = h(x * 2)
      self.assertEqual(jax.typeof(out).sharding.spec, P())
      return out

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P()))
    self.assertArraysEqual(out, np_op(np_inp * 2))

  @parameterized.named_parameters(
      ('max', jnp.max, np.max),
      ('min', jnp.min, np.min),
      ('sum', jnp.sum, np.sum),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2, 2), ('m', 'x', 'y'))
  def test_partial_manual(self, jnp_op, np_op, mesh):
    np_inp = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    arr = jax.device_put(np_inp, P('m', 'x', 'y'))

    @jax.jit
    @jax.shard_map(out_specs=P('m'), axis_names={'m'})
    def f(a):
      self.assertEqual(jax.typeof(a).sharding.spec, P(None, 'x', 'y'))
      self.assertEqual(a.aval.mat.varying, {'m'})
      y = jnp_op(a * 2, axis=1)
      self.assertEqual(jax.typeof(y).sharding.spec, P(None, 'y'))
      self.assertEqual(y.aval.mat.varying, {'m'})
      return y

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('m', 'y')))
    self.assertArraysEqual(out, np_op(np_inp * 2, axis=1))

  @parameterized.named_parameters(
      ('max', jnp.max, np.max),
      ('min', jnp.min, np.min),
      ('sum', jnp.sum, np.sum),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh(
      (2, 2, 2), ('m', 'e', 'a'),
      axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Auto))
  def test_manual_explicit_auto(self, jnp_op, np_op, mesh):
    np_inp = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('m', 'e', 'a')))

    @jax.jit
    @jax.shard_map(out_specs=P(None), axis_names={'m'})
    def f(a):
      self.assertEqual(jax.typeof(a).sharding.spec, P(None, 'e', None))
      self.assertEqual(a.aval.mat.varying, {'m'})
      y = jnp_op(a, axis=(1, 2))
      self.assertEqual(jax.typeof(y).sharding.spec, P(None))
      self.assertEqual(y.aval.mat.varying, {'m'})
      return lax.psum(y, 'm')

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None)))
    expected = np_op(np_inp[:2], axis=(1, 2)) + np_op(np_inp[2:], axis=(1, 2))
    self.assertArraysEqual(out, expected)

  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_out_sharding(self, mesh):
    np_inp = np.arange(32, dtype=np.float32).reshape(4, 8)
    arr = jax.device_put(np_inp, P('x', 'y'))
    u_spec = P('y', unreduced={'x'}, unreduced_kind=UnreducedKind.max)

    @jax.jit
    def f(x):
      y_nat = lax.reduce_sum(x, axes=(0,), out_sharding=P('y'))
      self.assertEqual(jax.typeof(y_nat).sharding.spec, P('y'))
      y_reshard = lax.reduce_sum(x, axes=(0,), out_sharding=P(None))
      self.assertEqual(jax.typeof(y_reshard).sharding.spec, P(None))
      y_scatter = lax.reduce_sum(
          jax.reshard(x, P('x', None)), axes=(0,), out_sharding=P('x'))
      self.assertEqual(jax.typeof(y_scatter).sharding.spec, P('x'))
      y_ur = lax.reduce_max(x, axes=(0,), out_sharding=u_spec)
      self.assertEqual(jax.typeof(y_ur).sharding.spec, u_spec)
      return y_nat, y_reshard, y_scatter, y_ur

    y_nat, y_reshard, y_scatter, y_ur = f(arr)
    self.assertEqual(y_nat.sharding, NamedSharding(mesh, P('y')))
    self.assertArraysEqual(y_nat, np.sum(np_inp, axis=0))
    self.assertEqual(y_reshard.sharding, NamedSharding(mesh, P(None)))
    self.assertArraysEqual(y_reshard, np.sum(np_inp, axis=0))
    self.assertEqual(y_scatter.sharding, NamedSharding(mesh, P('x')))
    self.assertArraysEqual(y_scatter, np.sum(np_inp, axis=0))
    self.assertEqual(y_ur.sharding, NamedSharding(mesh, u_spec))

  @parameterized.parameters(
      ((4, 8), P('x', None), (0,), P('x')),
      ((2, 2, 8), P('x', 'y', None), (0, 1), P(('x', 'y'))),
      ((2, 2, 8), P('x', 'y', None), (0, 1), P(('y', 'x'))),
      ((2, 2, 8), P('x', 'y', None), (0, 1), P('x')),
      ((2, 8), P('x', 'y'), (0,), P(('y', 'x'))),
      ((2, 2, 4, 8), P('x', 'y', None, None), (0, 1), P('x', 'y')),
  )
  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reduce_scatter(self, shape, in_spec, axes, out_spec, mesh):
    np_inp = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    arr = jax.device_put(np_inp, in_spec)

    def f(x):
      y = lax.reduce_sum(x, axes=axes, out_sharding=out_spec)
      self.assertEqual(jax.typeof(y).sharding.spec, out_spec)
      return y

    f_jit = jax.jit(f)
    lowered_text = f_jit.lower(arr).as_text()
    self.assertIn('sdy.manual_computation', lowered_text)
    self.assertIn('reduce_scatter', lowered_text)

    out = f_jit(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))
    self.assertArraysEqual(out, np.sum(np_inp, axis=axes))

  @config.shard_map_partitioning(True)
  @jtu.with_explicit_mesh((1, 2), ('x', 'y'))
  def test_size_one_axis(self, mesh):
    np_inp = np.arange(16, dtype=np.float32).reshape(4, 4)
    arr = jax.device_put(np_inp, P('x', 'y'))

    @jax.jit
    def f(x):
      y0 = jnp.min(x, axis=0)
      self.assertEqual(jax.typeof(y0).sharding.spec, P('y'))
      y1 = jnp.min(x, axis=1)
      self.assertEqual(jax.typeof(y1).sharding.spec, P('x'))
      return y0, y1, jnp.min(x)

    y0, y1, y01 = f(arr)
    self.assertEqual(y0.sharding, NamedSharding(mesh, P('y')))
    self.assertArraysEqual(y0, np.min(np_inp, axis=0))
    self.assertEqual(y1.sharding, NamedSharding(mesh, P('x')))
    self.assertArraysEqual(y1, np.min(np_inp, axis=1))
    self.assertArraysEqual(y01, np.min(np_inp))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
