import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import time
import subprocess
import csv

app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
# mw.resize(800,800)



win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')
pg.setConfigOptions(antialias=True)

p1 = win.addPlot()
xdata = {i:[] for i in xrange(9)}
ydata = {i:[] for i in xrange(9)}
curve1 = p1.plot([],[])
colors = ['b','g','r','c','m','y','k','b','g']
p1.addLegend()
ch = [p1.plot(xdata[i],ydata[i],pen=colors[i],thickness=2,name='Channel %d'%(i)) for i in xrange(9)]

# legend.setParentItem(p1)
ptr1 = 0


reader = csv.reader(open('datatest.csv', 'rU'))
def update1():
    try:
    	row = reader.next()
    except StopIteration:
    	return
    channel, curtime, resistance = row[0], row[1], row[2].split(' ')[0]
    index, curtime, resistance = int(channel), int(curtime), float(resistance)
    data_x = xdata[index]
    data_y = ydata[index]
    if len(data_y) < 50:
    	data_x.append(curtime)
    	data_y.append(resistance)
    else:
    	data_y[:-1] = data_y[1:]  # shift data in the array one sample left
                            # (see also: np.roll)
    	data_y[-1] = resistance
    	data_x[:-1] = data_x[1:]
    	data_x[-1] = curtime

    curve = ch[index]
    curve.setData(data_x,data_y)
    

def update():
	update1()
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()