from keras import backend as K
import tensorflow as tf
from tensorflow.python import saved_model
from tensorflow.python.saved_model.signature_def_utils_impl import (
    build_signature_def, predict_signature_def
)

from generators.pascal import PascalVocGenerator
from model import efficientdet
import shutil
import os


phi = 2
export_path = 'saved_model'
weighted_bifpn = False

generator = PascalVocGenerator(
    'datasets/VOC2007',
    'test',
    phi=phi,
    shuffle_groups=False,
    skip_truncated=False,
    skip_difficult=True,
)
model_path = 'checkpoints/2019-12-12/csv_29_1.0310_0.8090.h5'
num_classes = 2 #generator.num_classes()
classes = ["Issue", "Expiry"] #list(generator.classes.keys())
score_threshold = 0.5

model, prediction_model = efficientdet(phi=phi,
                                       weighted_bifpn=weighted_bifpn,
                                       num_classes=num_classes,
                                       score_threshold=score_threshold)
prediction_model.load_weights(model_path, by_name=True)

print('Output layers', [o.name[:-2] for o in model.outputs])
print('Input layer', model.inputs[0].name[:-2])
if os.path.isdir(export_path):
    shutil.rmtree(export_path)
builder = saved_model.builder.SavedModelBuilder(export_path)

signature = predict_signature_def(
    inputs={'input_1': model.input},
    outputs={
        'regression/concat': model.outputs[0],
        'classification/concat': model.outputs[1]#,
        #3'output3': model.outputs[2]
    }
)

sess = K.get_session()
builder.add_meta_graph_and_variables(sess=sess,
                                     tags=[saved_model.tag_constants.SERVING],
                                     signature_def_map={'predict': signature})
builder.save()
