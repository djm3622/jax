# Copyright 2024 The JAX Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from absl.testing import absltest

import jax
from jax import numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax import linalg as lax_linalg
from jax._src.lib import version as jaxlib_version  # pylint: disable=g-importing-member

import numpy as np

config.parse_flags_with_absl()
jax.config.update("jax_enable_x64", True)


class CholeskyUpdateTest(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cuda"]):
      self.skipTest("Only works on a CUDA GPU")
    if jaxlib_version < (0, 4, 28):
      self.skipTest("Requires jaxlib 0.4.28 or newer")

  def testUpperOnes(self):
    r_upper = jnp.triu(jnp.ones((1024, 1024)))
    w = jnp.arange(1, 1024 + 1).astype(jnp.float64)
    updated = lax_linalg.cholesky_update(r_upper, w)

    new_matrix = r_upper.T @ r_upper + jnp.outer(w, w)
    new_cholesky = jnp.linalg.cholesky(new_matrix, upper=True)
    jtu._assert_numpy_allclose(updated, new_cholesky)

  @jtu.sample_product(
      shape=[
          (128, 128),
          (1024, 1024),
          (4096, 4096),
      ],
  )
  def testRandomMatrix(self, shape):
    rng = jtu.rand_default(self.rng())
    a = rng(shape, np.float64)
    pd_matrix = jnp.array(a.T @ a)
    old_cholesky = jnp.linalg.cholesky(pd_matrix, upper=True)

    w = rng((shape[0],), np.float64)
    w = jnp.array(w)

    new_matrix = pd_matrix + jnp.outer(w, w)
    new_cholesky = jnp.linalg.cholesky(new_matrix, upper=True)
    updated = lax_linalg.cholesky_update(old_cholesky, w)

    jtu._assert_numpy_allclose(updated, new_cholesky, atol=1e-6)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
