# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""XManager launcher for clowder.

Usage:

xmanager launch clowder/launcher.py -- \
  --xm_wrap_late_bindings \
  [--image_path=gcr.io/path/to/image/tag] \
  [--platform=gpu]
"""
import itertools

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local

FLAGS = flags.FLAGS
flags.DEFINE_string('image_path', None, 'Image path.')
flags.DEFINE_string('platform', 'cpu', 'cpu/gpu/tpu.')
flags.DEFINE_integer('cores', 1, 'Number of cores. Use 8 if platform==tpu.')


def main(_):
  with xm_local.create_experiment(experiment_title='ppo_atari_multigpu') as experiment:
    if FLAGS.image_path:
      spec = xm.Container(image_path=FLAGS.image_path)
    else:
      # Package the current directory that this script is in.
      spec = xm.PythonContainer(
          path='.',
          # This base_image is experimental and works with cpu/gpu/tpu.
          # https://cloud.google.com/ai-platform/deep-learning-containers/docs/choosing-container
          base_image='gcr.io/deeplearning-platform-release/pytorch-xla.1-8',
          entrypoint=xm.ModuleName('ppo_atari_multigpu'),
      )
      
    xm_local.Local(
      xm.JobRequirements(local_gpu=1)
    )
    
    [executable] = experiment.package(
        [
            xm.Packageable(
                executable_spec=spec,
                executor_spec=xm_local.Local.Spec(),
                args={'platform': FLAGS.platform},
            ),
        ]
    )

    num_minibatcheses = [4, 5]
    learning_rates = [2.5e-4, 2.6e-4]
    trials = list(
        dict([('num_minibatches', nm), ('learning_rate', lr)])
        for (nm, lr) in itertools.product(num_minibatcheses, learning_rates)
    )

    requirements = xm.JobRequirements()
    if FLAGS.platform == 'gpu':
      requirements = xm.JobRequirements(local_gpu=2)
    elif FLAGS.platform == 'tpu':
      requirements = xm.JobRequirements(tpu_v3=8)
    for hyperparameters in trials:
      jobs = {}
      jobs['coordinator'] = xm.Job(
          executable=executable,
          executor=xm_local.Local(requirements),
          args=hyperparameters,
      )
      experiment.add(xm.JobGroup(**jobs))
      break
    print('Waiting for async launches to return values...')
  print('Launch completed and successful.')


if __name__ == '__main__':
  app.run(main)