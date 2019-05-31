from visdom import Visdom
import numpy as np
import math
import os.path

class visdom_painter():
    def __init__(self, port = 28100):
        self.viz = Visdom(port = port)
        self.win_list = {}
    
    def create_window(self, initial_x, initial_y, window_name, title,
                    legends = None):
        window_opt = {'title': title}
        if legends != None:
            window_opt['legend'] = legends
        window = self.viz.line(
                X = initial_x,
         	    Y = initial_y,
                win = window_name, 
		        opts = window_opt,
        )
        self.win_list[window_name] = window
    
    def decorate_html(self, key, value):
        return "<div>" + str(key) + ": " + str(value) + "</div>"

    def add_para(self, arg_dict):
        para_sentence = '<div style="font-size: 20px; font-weight: 900">Parameter</div>'
        for key, value in arg_dict.items():
            para_sentence += self.decorate_html(key, value)
        self.viz.text(para_sentence)
    
    def update_data(self, window_name, x, y):
        window = self.win_list.get(window_name, None)
        self.viz.line(
            X = x,
            Y = y,
            win = window,
            update='append',
        )
