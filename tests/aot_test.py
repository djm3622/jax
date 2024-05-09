# Copyright 2021 The JAX Authors.
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
"""Tests for AOT compilation."""

import contextlib
import glob
import logging
import os
import tempfile
import unittest
from absl.testing import absltest
import jax
from jax._src import core
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.lib import xla_client as xc
from jax._src.lib import xla_extension
from jax.experimental import topologies
from jax.experimental.pjit import pjit
from jax.experimental.serialize_executable import (
    deserialize_and_load,
    serialize,
)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

jax.config.parse_flags_with_absl()

prev_xla_flags = None

with contextlib.suppress(ImportError):
  import pytest
  pytestmark = pytest.mark.multiaccelerator


class JaxAotTest(jtu.JaxTestCase):

  @jtu.run_on_devices('tpu', 'gpu')
  def test_pickle_pjit_lower(self):
    def fun(x):
      return x * x

    with jax.sharding.Mesh(np.array(jax.devices()), ('data',)):
      lowered = pjit(
          fun, in_shardings=P('data'), out_shardings=P(None, 'data')
      ).lower(core.ShapedArray(shape=(8, 8), dtype=np.float32))

    def verify_serialization(lowered):
      serialized, in_tree, out_tree = serialize(lowered.compile())
      compiled = deserialize_and_load(serialized, in_tree, out_tree)
      self.assertEqual(compiled.as_text(), lowered.compile().as_text())

    verify_serialization(lowered)
    verify_serialization(jax.jit(lambda x: x * x).lower(np.arange(100)))
    verify_serialization(
        jax.pmap(lambda x: x * x).lower(
            np.zeros((len(jax.devices()), 4), dtype=np.float32)))

  def test_topology_pjit_serialize(self):
    try:
      aot_topo = topologies.get_topology_desc(
          platform=jax.devices()[0].platform
      )
    except NotImplementedError:
      raise unittest.SkipTest('PJRT Topology not supported')

    if jtu.TEST_WITH_PERSISTENT_COMPILATION_CACHE.value:
      raise unittest.SkipTest('Compilation caching not yet supported.')

    @jax.jit
    def fn(x):
      return x * x

    def lower_and_load(mesh):
      s = jax.sharding.NamedSharding(mesh, P('x', 'y'))
      x_shape = jax.ShapeDtypeStruct(
          shape=(16, 16),
          dtype=jnp.dtype('float32'),
          sharding=s)
      lowered = fn.lower(x_shape)
      serialized, in_tree, out_tree = serialize(lowered.compile())
      compiled = deserialize_and_load(serialized, in_tree, out_tree)
      return compiled

    ref_topo = topologies.get_attached_topology()
    n = max(1, len(ref_topo.devices) // 2)
    mesh_shape = (len(ref_topo.devices) // n, n)

    ref_mesh = topologies.make_mesh(ref_topo, mesh_shape, ('x', 'y'))
    aot_mesh = topologies.make_mesh(aot_topo, mesh_shape, ('x', 'y'))
    self.assertEqual(
        lower_and_load(ref_mesh).as_text(), lower_and_load(aot_mesh).as_text()
    )

  def test_get_topology_from_devices(self):
    try:
      aot_topo = topologies.get_topology_desc(
          platform=jax.devices()[0].platform
      )
    except NotImplementedError:
      raise unittest.SkipTest('PJRT Topology not supported')

    topo = xc.get_topology_for_devices(aot_topo.devices)
    self.assertEqual(
        topo.platform_version, aot_topo.devices[0].client.platform_version
    )

  @jtu.run_on_devices('tpu')
  def test_tpu_profiler_registered_get_topology_from_devices(self):
    if not xla_bridge.using_pjrt_c_api():
      raise unittest.SkipTest('Profiler plugin only works with plugin backend')

    try:
      _ = topologies.get_topology_desc(
          topology_name='fake_topology',
          platform='tpu',
      )
    except xla_extension.XlaRuntimeError:
      logging.info('Expected to fail to get topology')

    with tempfile.TemporaryDirectory() as tmpdir:
      try:
        jax.profiler.start_trace(tmpdir)
        jax.pmap(lambda x: jax.lax.psum(x + 1, 'i'), axis_name='i')(
            jnp.ones(jax.local_device_count())
        )
      finally:
        jax.profiler.stop_trace()

      proto_path = glob.glob(
          os.path.join(tmpdir, '**/*.xplane.pb'), recursive=True
      )
      self.assertLen(proto_path, 1)
      with open(proto_path[0], 'rb') as f:
        proto = f.read()
      # Sanity check that serialized proto contains host, and Python traces
      # without deserializing.
      self.assertIn(b'/host:CPU', proto)
      if jtu.test_device_matches(['tpu']):
        self.assertNotIn(b'/device:TPU', proto)
      self.assertIn(b'pxla.py', proto)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
