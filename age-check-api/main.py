import flask
from flask import Flask, jsonify, request
from flask_restful import Api, Resource, abort
import requests
from PIL import Image
from io import BytesIO

from transformers import AutoImageProcessor, AutoModelForImageClassification

app = Flask(__name__)

@app.route('/test', methods=['POST'])
def create_task():
    if not request.json or not 'title' in request.json:
        abort(400)

    r = requests.get('{0}?raw=true'.format('https://github.com/dchen236/FairFace/blob/master/detected_faces/race_Asian_face0.jpg'))
    im = Image.open(BytesIO(r.content))

    processor = AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier")
    model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")

    # Transform our image and pass it through the model
    inputs = processor(im, return_tensors='pt')
    output = model(**inputs)

    # Predicted Class probabilities
    proba = output.logits.softmax(1)

    # Predicted Classes
    preds = proba.argmax(1)

    return jsonify({'task': preds}), 201

if __name__ == '__main__':
    app.run(debug=True)


# Get example image from official fairface repo + read it in as an image
# r = requests.get('{0}?raw=true'.format(url))
# im = Image.open(BytesIO(r.content))

# Init model, transforms
# processor = AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier")
# model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")
#
# # Transform our image and pass it through the model
# inputs = processor(im, return_tensors='pt')
# output = model(**inputs)
#
# # Predicted Class probabilities
# proba = output.logits.softmax(1)
#
# # Predicted Classes
# preds = proba.argmax(1)
# print(str(preds))
# return str(preds)