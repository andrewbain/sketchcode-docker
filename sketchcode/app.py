import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
from PIL import Image
from io import BytesIO
import base64
import re
import time
from model import SketchModel
from config import Config

# define the app
app = Flask(__name__)
CORS(app, resources={r"/api": {"origins": "*"}}) # needed for cross-domain requests, allow everything by default
app.config['CORS_HEADERS'] = 'Content-Type'



# load the model
model = SketchModel(Config)


# API route
@app.route('/api', methods=['POST'])
def api():
    """API function
    All model-specific logic to be defined in the get_model_api()
    function
    """
    data = json.loads(request.data)

    image_data = re.sub('^data:image/.+;base64,', '', data['image'])
    image_data = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data))
    path = os.path.dirname(sys.argv[0]) + '/images/temp_folder/' + 'input_{}.jpg'.format(int(time.time()))
    image.save(path)


    prediction = model.predict(path)
    return json.dumps(prediction)


@app.route('/')
def index():
    return "Sketchcode index"

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally.
    app.run(host='0.0.0.0')
