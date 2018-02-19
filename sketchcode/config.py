import os
from keras.utils.data_utils import get_file
from keras.models import Model, Sequential, load_model

class Config():
    def __init__(self):
        self.num_classes = len(self.load_classes())
        self.model = self.load_model()

    def load_classes(self):
        classes = [{'id':0, 'directory': 'nav_standard', 'type':'nav_h', 'markup': 'logo / menuitem / menuitem / menuitem / menuitem'},
                    {'id':1, 'directory': 'full_width_image_title', 'type': 'row', 'markup': 'image_full,title,center,center'},
                   {'id': 2, 'directory': 'full_width_image_button', 'type': 'row',
                    'markup': 'image_full,button,center,center'},
                    {'id': 3, 'directory': 'image_text', 'type': 'row', 'markup': 'image_full / transparent,text'},
                    {'id': 4, 'directory': 'text_image', 'type': 'row', 'markup': 'transparent,text / image_full'}]
        return classes

    def load_model(self):
        model_file = get_file('custom-model-latest.h5', 'https://s3-eu-west-1.amazonaws.com/andrewbain/sketchcode/models/custom-model-latest.h5')
        model = load_model(model_file)
        return model