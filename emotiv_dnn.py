from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import json
import iris_data

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--train_steps', default=2500, type=int,
                    help='number of training steps')

    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[48,48,48,48,48,48],
        # The model must choose between 3 classes.
        n_classes=2,
        activation_fn=tf.nn.crelu,
        model_dir="emotiv_neuron_model_709"
    )

    # Generate predictions from the model
    resultados = {}
    resultados["predicciones"] = []
    prediccion = ''
    confiabilidad =  0.0
    with open('pregunta.json') as json_file:
        preguntas = json.load(json_file)
        for data in preguntas['preguntas']:
            predict_x = {
                'AF3': [data["AF3"]],
                'F3': [data["F3"]],
                'AF4': [data["AF4"]],
                'F4': [data["F4"]],
                'sexo': [data["sexo"]],
                'cief': [data["cief"]],
                'hare': [data["hare"]],
                'pebl': [data["pebl"]],
                'edad': [data["edad"]],
            }
            predictions = classifier.predict(
                input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                        labels=None,
                                                        batch_size=100))
            for pred_dict in predictions:
                class_id = pred_dict['class_ids'][0]
                probability = pred_dict['probabilities'][class_id]
                prediccion = iris_data.VERACIDAD[class_id]
                confiabilidad = 100*probability
                resultados["predicciones"].append({
                    "prediccion": iris_data.VERACIDAD[class_id],
                    "confiabilidad" : 100*probability
                    })
                print (iris_data.VERACIDAD[class_id])
                print (100*probability)
    with open('resultados.json', 'w') as outfile:  
            json.dump(resultados, outfile)
    return prediccion, confiabilidad

if __name__ == '__main__':
    tf.app.run(main)
