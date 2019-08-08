try:
  import unzip_requirements
except ImportError:
  pass


import json
import boto3 


import tensorflow as tf
from tensorflow.python.platform import gfile
#from tensorflow.python.keras.preprocessing import image
from PIL import Image

import numpy as np
import base64
import io

    
    
def run_classify_image(img):
    
    f = gfile.FastGFile("tf-models/tf_model.pb", 'rb')
    graph_def = tf.GraphDef()
   # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()

    sess = tf.Graph()
    with sess.as_default() as graph:
        tf.import_graph_def(graph_def)
        softmax_tensor = sess.get_tensor_by_name('import/activation_15_2/Softmax:0')

    with tf.Session(graph=graph) as sess:
        predictions = sess.run(softmax_tensor, {'import/conv2d_6_input_2:0': img})
         
    return predictions    
        


def singlepostHandler(event, context):
    body_txt = event['body']
    body_json = json.loads(body_txt)
    post = body_json["post_content"]



    response = {
        "statusCode": 200,
        "headers": {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
        "body": json.dumps({'post': post})
    }
    
    return response

