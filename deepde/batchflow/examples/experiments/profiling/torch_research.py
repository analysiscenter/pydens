""" Test research with torch model """

import sys

sys.path.append("../../..")
from batchflow import Pipeline, B, C, V, D
from batchflow.opensets import MNIST
from batchflow.models.torch import VGG16
from batchflow.research import Research

BATCH_SIZE = 64

model_config = {
    'inputs/images/shape': (1, 28, 28),
    'inputs/labels/classes': D('num_classes'),
    'initial_block/inputs': 'images',
    'body/block/layout': 'cna',
    'device': C('device') # it's technical parameter for TFModel
}

mnist = MNIST()
train_root = mnist.train.p.run(BATCH_SIZE, shuffle=True, n_epochs=None, lazy=True)
test_root = mnist.test.p.run(BATCH_SIZE, shuffle=True, n_epochs=1, lazy=True)

train_template = (Pipeline()
    .init_variable('loss', init_on_each_run=list)
    .init_variable('accuracy', init_on_each_run=list)
    .init_model('dynamic', VGG16, 'conv', config=model_config)
    .to_array(channels='first', dtype='float32')
    .train_model('conv', B('images'), B('labels'),
                fetches='loss', save_to=V('loss', mode='w'))
)

test_template = (Pipeline()
    .init_variable('predictions')
    .init_variable('metrics', init_on_each_run=None)
    .import_model('conv', C('import_from'))
    .to_array(channels='first', dtype='float32')
    .predict_model('conv', B('images'),
                   fetches='predictions',
                   save_to=V('predictions'))
    .gather_metrics('class', targets=B('labels'), predictions=V('predictions'),
                    fmt='logits', axis=-1, save_to=V('metrics', mode='a'))
)

train_ppl = train_root + train_template
test_ppl = test_root + test_template

research = (Research()
    .add_pipeline(train_ppl, variables='loss', name='train')
    .add_pipeline(test_ppl, name='test', run=True, execute='%100', import_from='train')
    .get_metrics('test', 'metrics', 'accuracy', returns='accuracy', execute='%100')
)

n_workers = 1 if len(sys.argv) <= 1 else int(sys.argv[1])
gpu_list = [2, 4, 5, 6]

research.run(n_reps=8, n_iters=1000, workers=n_workers, name='torch_research_'+str(n_workers), gpu=gpu_list[:n_workers])
