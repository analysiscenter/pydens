""" Test tf model """

import sys

sys.path.append("../../..")
from batchflow import B, V, D
from batchflow.opensets import MNIST
from batchflow.models.tf import VGG16

BATCH_SIZE = 64

model_config = {
    'inputs/images/shape': (28, 28, 1),
    'inputs/labels/classes': D('num_classes'),
    'initial_block/inputs': 'images',
    'body/block/layout': 'cna',
    'device': 'gpu:2'
}

mnist = MNIST()

train_ppl = (mnist.train.p
    .init_variable('loss', init_on_each_run=list)
    .init_variable('accuracy', init_on_each_run=list)
    .init_model('dynamic', VGG16, 'conv', config=model_config)
    .to_array()
    .train_model('conv',
                 images=B('images'), labels=B('labels'),
                 fetches='loss',
                 save_to=V('loss', mode='w'))
    .run(BATCH_SIZE, shuffle=True, n_epochs=1, lazy=True))


test_ppl = (mnist.test.p
    .init_variable('predictions')
    .init_variable('metrics', init_on_each_run=None)
    .import_model('conv', train_ppl)
    .to_array()
    .predict_model('conv',
                   images=B('images'),
                   fetches='predictions',
                   save_to=V('predictions'))
    .gather_metrics('class', targets=B('labels'), predictions=V('predictions'),
                    fmt='logits', axis=-1, save_to=V('metrics', mode='a'))
    .run(BATCH_SIZE, shuffle=True, n_epochs=1, lazy=True))

train_ppl.run()
test_ppl.run()
