import os

class Config():
    def __init__(self):
        self.num_classes = len(self.load_classes())

    def load_classes(self):
        classes = [{'id':0, 'directory': 'nav_standard', 'type':'nav_h', 'markup': 'logo / menuitem / menuitem / menuitem / menuitem'},
                    {'id':1, 'directory': 'full_width_image_title', 'type': 'row', 'markup': 'image_full,title,center,center'},
                   {'id': 2, 'directory': 'full_width_image_button', 'type': 'row',
                    'markup': 'image_full,button,center,center'},
                    {'id': 3, 'directory': 'image_text', 'type': 'row', 'markup': 'image_full / transparent,text'},
                    {'id': 4, 'directory': 'text_image', 'type': 'row', 'markup': 'transparent,text / image_full'}]
        return classes