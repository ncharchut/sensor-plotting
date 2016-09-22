# pylint: disable=wrong-import-position, unused-import

import csv
from datetime import datetime
from collections import deque
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
# import itertools
import Tkinter    #   may need these uncommented for app build
import FileDialog




colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'darkgreen']
font = FontProperties()
font.set_family('serif')
matplotlib.rc('font', family='serif')
