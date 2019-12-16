import sys
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python import saved_model
from tensorflow.python.saved_model.signature_def_utils_impl import (
    build_signature_def, predict_signature_def
)

from generators.pascal import PascalVocGenerator
from model import efficientdet
import shutil
import os



K.set_learning_phase(0)
K.set_image_data_format('channels_last')

FROZEN_FOLDER= './models/frozen'
FROZEN_GRAPH = 'frozen_model.pb'
SAVED_FOLDER = './models/saved_model'


phi = 1
weighted_bifpn = False
model_path = './models/keras/pascal_05_0.6283_1.1975_0.8029.h5'
num_classes = 20 #generator.num_classes()

score_threshold = 0.5
model, prediction_model = efficientdet(phi=phi,
                                       weighted_bifpn=weighted_bifpn,
                                       num_classes=num_classes,
                                       score_threshold=score_threshold)

# print('Output layers', [o.name[:-2] for o in model.outputs])
# print('Input layer', model.inputs[0].name[:-2])               

# FROZEN MODEL
output_node_names = [o.name[:-2] for o in model.outputs]
input_node_names = [i.name[:-2] for i in model.inputs]

print(input_node_names)
print(output_node_names)
sess = K.get_session()
try:
    frozen_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names)    
    graph_io.write_graph(frozen_graph, FROZEN_FOLDER, FROZEN_GRAPH, as_text=False)
    print(f'Frozen graph ready for inference/serving at {FROZEN_FOLDER}/{FROZEN_GRAPH}')
except:
    print('Error Occured')


# SAVED MODEL
builder = saved_model.builder.SavedModelBuilder(SAVED_FOLDER)
signature = predict_signature_def(
    inputs={'input_1': model.input},
    outputs={
        'regression/concat': model.outputs[0],
        'classification/concat': model.outputs[1]
    }
)

sess = K.get_session()
builder.add_meta_graph_and_variables(sess=sess,
                                     tags=[saved_model.tag_constants.SERVING],
                                     signature_def_map={'predict': signature})
builder.save()




#-------------------------------------------------------------------------------------------------------------
#! EXPORT SAVED MODEL FROM FROZEN


# OUTPUT_SERVABLE_FOLDER = sys.argv[2]
# INPUT_TENSOR = sys.argv[3]
# builder = tf.saved_model.builder.SavedModelBuilder(OUTPUT_SERVABLE_FOLDER)

# with tf.gfile.GFile(f'{OUTPUT_FOLDER}/{OUTPUT_GRAPH}', "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())

# sigs = {}
# OUTPUT_TENSOR = output_node_names
# with tf.Session(graph=tf.Graph()) as sess:
#     tf.import_graph_def(graph_def, name="")
#     g = tf.get_default_graph()
#     inp = g.get_tensor_by_name(INPUT_TENSOR)
#     out = g.get_tensor_by_name(OUTPUT_TENSOR[0] + ':0')

#     sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
#         tf.saved_model.signature_def_utils.predict_signature_def(
#             {"input": inp}, {"outout": out})

#     builder.add_meta_graph_and_variables(sess,
#                                          [tag_constants.SERVING],
#                                          signature_def_map=sigs)
#     try:
#         builder.save()
#         print(f'Model ready for deployment at {OUTPUT_SERVABLE_FOLDER}/saved_model.pb')
#         print('Prediction signature : ')
#         print(sigs['serving_default'])
#     except:
#         print('Error Occured, please checked frozen graph')