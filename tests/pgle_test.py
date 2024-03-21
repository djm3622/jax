# Copyright 2023 The JAX Authors.
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
import glob
import logging
import math
import os
import tempfile

from absl.testing import absltest
import jax
from jax._src import config
from jax._src import pjit
from jax._src import test_util as jtu
from jax._src import api
from jax.experimental import profiler as exp_profiler
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
from jax._src import compilation_cache as cc
from jax._src.lib import xla_extension_version
import numpy as np

from jax.experimental.serialize_executable import (
    deserialize_and_load,
    serialize,
)

jax.config.parse_flags_with_absl()


@jtu.pytest_mark_if_available('multiaccelerator')
class PgleTest(jtu.JaxTestCase):

  def testAutoPgle(self):
    if xla_extension_version <= 262:
      return

    mesh = jtu.create_global_mesh((2,), ('x',))

    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, PartitionSpec('x')),
        out_shardings=NamedSharding(mesh, PartitionSpec('x')),
    )
    def f(x):
      return x * 2

    shape = (16, 16)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
    expected = x * 2

    module_dict = pjit._most_recent_pjit_call_executable.weak_key_dict
    module_dict.clear()
    with config.pgle_profiling_runs(2):

      # Run 1
      with jtu.count_jit_and_pmap_compiles() as cache_miss_count:
        self.assertArraysEqual(f(x), expected)
      # after_first_usage = pxla._cached_compilation.cache_info()

      # Module should be compiled so compilation cache is not hit
      # Two modules are expected
      # One is the funtion f, the other one is multi slice module
      self.assertEqual(cache_miss_count[0], 2)
      self.assertLen(module_dict, 2)
      original_modules = [id(module) for module in module_dict.values()]

      # Run 2
      with jtu.count_jit_and_pmap_compiles() as cache_miss_count:
        self.assertArraysEqual(f(x), expected)

      # Second PGLE run should not recompile the module
      self.assertEqual(cache_miss_count[0], 0)

      # Two modules are expected
      # One is the funtion f, the other one is multi slice module
      self.assertLen(module_dict, 2)

      # Modules should not be recompiled
      modules = [id(module) for module in module_dict.values()]
      self.assertArraysEqual(modules, original_modules)

      # Run 3
      self.assertArraysEqual(f(x), expected)
      self.assertLen(module_dict, 2)

      # The module should be recompiled with FDO
      recompiled_modules = [id(module) for module in module_dict.values()]
      for module, orig_module in zip(recompiled_modules, original_modules):
        self.assertNotEqual(module, orig_module)

      # Run 4
      self.assertArraysEqual(f(x), expected)
      self.assertLen(module_dict, 2)

      # After recompilation the module should not be recompiled anymore
      modules = [id(module) for module in module_dict.values()]
      self.assertArraysEqual(modules, recompiled_modules)

  def testAutoPgleWithAot(self):
    if xla_extension_version <= 262:
      return

    @jax.jit
    def f(x):
      return x * 2

    x = jnp.arange(1)
    expected = x * 2

    f_lowered = f.lower(x)
    serialized, in_tree, out_tree = serialize(f_lowered.compile())
    compiled = deserialize_and_load(serialized, in_tree, out_tree)

    module_dict = pjit._most_recent_pjit_call_executable.weak_key_dict
    module_dict.clear()

    with config.pgle_profiling_runs(1):
      # Run 1
      self.assertArraysEqual(compiled(x), expected)
      self.assertEmpty(module_dict)

      # Run 2
      self.assertArraysEqual(compiled(x), expected)
      self.assertEmpty(module_dict)

  def setUp(self):
    super().setUp()
    cc.reset_cache()

  def tearDown(self):
    cc.reset_cache()
    super().tearDown()

  def testAutoPgleWithPersistentCache(self):
    if xla_extension_version <= 262:
      return

    @jax.jit
    def f(x):
      return x * 2

    x = jnp.arange(1)
    expected = x * 2

    module_dict = pjit._most_recent_pjit_call_executable.weak_key_dict
    module_dict.clear()

    profilers_dict = (
        pjit._most_recent_pjit_call_executable.weak_profile_session_runner_dict)
    with (config.enable_compilation_cache(True),
          config.raise_persistent_cache_errors(True),
          config.persistent_cache_min_entry_size_bytes(0),
          config.persistent_cache_min_compile_time_secs(0),
          config.pgle_profiling_runs(1),
          tempfile.TemporaryDirectory() as tmpdir):
      cc.set_cache_dir(tmpdir)
      # Run 1
      self.assertArraysEqual(f(x), expected)
      self.assertLen(module_dict, 1)
      original_modules = [id(module) for module in module_dict.values()]

      # Run 2
      self.assertArraysEqual(f(x), expected)

      recompiled_modules = [id(module) for module in module_dict.values()]
      # Modules should be recompiled
      self.assertNotEqual(recompiled_modules[0], original_modules[0])

      for profiler in profilers_dict.values():
        self.assertTrue(profiler.is_enabled())
        self.assertTrue(profiler.is_fdo_consumed())
      files_in_cache_directory = os.listdir(tmpdir)
      self.assertLen(files_in_cache_directory, 1)

      api.clear_caches()
      profilers_dict.clear()

      # Run 3, compilation cache should be hit, PGLE should be disabled
      self.assertArraysEqual(f(x), expected)

      self.assertLen(profilers_dict, 1)
      for profiler in profilers_dict.values():
        self.assertFalse(profiler.is_enabled())
        self.assertFalse(profiler.is_fdo_consumed())

  def testPassingFDOProfile(self):
    mesh = jtu.create_global_mesh((2,), ('x',))

    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, PartitionSpec('x')),
        out_shardings=NamedSharding(mesh, PartitionSpec('x')),
    )
    def f(x, y):
      return x @ y

    shape = (16, 16)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
    y = x + 1

    with config.pgle_profiling_runs(0):
      f_lowered = f.lower(x, y)
      compiled = f_lowered.compile()

    with tempfile.TemporaryDirectory() as tmpdir:
      jax.profiler.start_trace(tmpdir)
      compiled(x, y)
      jax.profiler.stop_trace()
      directories = glob.glob(os.path.join(tmpdir, 'plugins/profile/**/'))
      directories = [d for d in directories if os.path.isdir(d)]
      rundir = directories[-1]
      logging.info('rundir: %s', rundir)
      fdo_profile = exp_profiler.get_profiled_instructions_proto(rundir)

    if jtu.test_device_matches(['gpu']) and jtu.is_device_cuda():
      self.assertIn(b'custom', fdo_profile)

    logging.info('fdo_profile: %s', fdo_profile)
    # Test pass fdo_profile as compiler_options API works.
    f_lowered.compile(compiler_options={'fdo_profile': fdo_profile})


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
