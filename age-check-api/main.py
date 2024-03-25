import os
from re import findall
from flask import Flask, request
from requests import get
from PIL import Image
from io import BytesIO

from transformers import AutoImageProcessor, AutoModelForImageClassification

ASSETS_DIR = os.path.join(os.path.dirname(__file__))
app = Flask(__name__)

@app.route('/test', methods = ['GEt'])
def test():
    return "Hello World!"

@app.route('/adultcheck',methods = ['POST'])
def login():

    if 'file' not in request.files:
        url = request.form['imageurl']
        r = get('{0}?raw=true'.format(url))
        im = Image.open(BytesIO(r.content))
        print('no files')
    else:
        print('has file')
        file = request.files['file']
        im = Image.open(file)

    im = im.convert('RGB')

    processor = AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier")
    model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")

    # Transform our image and pass it through the model
    inputs = processor(im, return_tensors='pt')
    output = model(**inputs)

    # Predicted Class probabilities
    proba = output.logits.softmax(1)

    # Predicted Classes
    preds = proba.argmax(1)
    string = str(preds)

    nums = findall(r'\d+', string)
    nums = [int(i) for i in nums]
    adult = nums[0] > 2

    return str(adult)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)