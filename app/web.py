from flask import Flask

import os
import cv2
import time
import numpy as np

# from utils import get_config
from .LP_reg_src import LP_regconition

  
import os
import yaml
from easydict import EasyDict as edict

from flask import Flask, render_template

import os
import logging
import traceback
import time

from utils import get_config

# set up config
cfg = get_config()
cfg.merge_from_file('./configs/service.yaml')

LOG_PATH = cfg.SERVICE.LOG_PATH
HOST = cfg.SERVICE.SERVICE_IP
PORT = cfg.SERVICE.PORT

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
@app.route('/home')
def view_home():
    return render_template('index.html')

@app.route('/regconition_image', method=['POST', 'GET'])
def view_regconition():
    

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=False)
