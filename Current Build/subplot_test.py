

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from pprint import pprint
from SensorLive import SensorLive
from SensorFinal import SensorFinal
import csv
import time
import telnetlib
import subprocess


def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        # ax.text(0.5, 0.5, "ax%d" % (i+1)) #, va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

fig = plt.figure(figsize=(18, 12), dpi=80)
gs = gridspec.GridSpec(4,4)

axs = [plt.subplot(gs[:-1,1:]),
       # plt.subplot(gs[0,0]),
       # plt.subplot(gs[1,0]),
       # plt.subplot(gs[2,0]),
       plt.subplot(gs[-1,0])]
       # plt.subplot(gs[-1,1]),
       # plt.subplot(gs[-1,2]),
       # plt.subplot(gs[-1,3])]
# ax9 = plt.subplot(gs[-1,-1])

# for i in xrange():
    # axs[i].locator_params(axis='x',nbins=3)



sensors = [SensorLive('test %d' % (i), fig=fig, ax=axs[i]) for i in xrange(1)]
# sensors[0].speed = 1
# leg = axs[0].legend()
# sensors[1].speed = 6
# sensor_dict = {0:sensors[0], 1:sensors[1]}
# sensor = sensors[0]

for sensor in sensors:
    sensor.leg.remove()

lines = [#axs[0].plot([1,2,3],[1,2,3],c='red'),
       sensors[0].lines,
       axs[1].plot([1,2,3],[4,5,6],c='green')]
       # sensors[1].lines]
       # axs[2].plot([1,2,3],[6,5,4],c='pink', label='ch0'),
       # sensors[2].lines,
       # axs[3].plot([1,2,3],[8,9,10],c='orange', label='ch0'),
       # # sensors[3].lines,
       # axs[4].plot([1,2,3],[3,2,1],c='blue', label='ch0'),
       # axs[5].plot([1,2,3],[10,9,8],c='yellow', label='ch0'),
       # axs[6].plot([1,2,3],[6,5,6],c='purple', label='ch0'),
       # axs[7].plot([1,2,3],[4,5,4],c='brown', label='ch0')]
# ax9.plot([1,2,3],[1,5,1],c='black')
def legend_init(ax, i):
    """
    Initializes the legend, establishes connection between legend lines
    and respective plot lines.
    Returns:
        LegendItem : legend of the graph
        dict (2DLine object): legend lines to plot lines
    """
    leg = ax.legend(ncol=2, fancybox=True, shadow=True, loc='upper left',
                         framealpha=0.5)
    leg.get_frame().set_alpha(0.4)
    leg.get_frame().set_facecolor('LightGreen')

    lined = {}
    for legline, origline in zip(leg.get_lines(), lines[i]):
        legline.set_picker(7)  # 7 pts tolerance
        lined[legline] = origline

    return leg

plt.suptitle("Test Subplots")
# make_ticklabels_invisible(plt.gcf())
global leg
leg = legend_init(axs[0], 0)

def changeColor(ax):
    print ax.get_position()
    for loc, spine in ax.spines.iteritems():
        if spine.get_ec() == (0, 0, 0, 1):
            spine.set_color('red')
            axes_selected = ax
        else:
            spine.set_color('black')

    plt.draw()




def switchAxes(index):
    if index == 0:
        return
    # try:    
    #     sensor_dict[0].leg.set_visible(False)
    # except KeyError:
    #     pass
    # try:
    #     sensor_dict[index].leg.set_visible(True)
    # except KeyError:
    #     pass
    global leg
    leg.remove()
    loc_f = axs[0].get_position() # will be attribute
    loc_0 = axs[index].get_position()
    axs[0].set_position(loc_0)
    axs[index].set_position(loc_f)
    axs[0], axs[index] = axs[index], axs[0]
    leg = legend_init(axs[0],0)
    # if 0 in sensors:
    #     sensors[0].blitting = False
    #     for line in sensors[0].lines.values():
    #         line.animated = False
    # if index in sensors:
    #     sensors[index].blitting = True
    #     for line in sensors[index].lines.values():
    #         line.animated = True
    # sensors[0], sensors[index] = sensors[index], sensors[0]
        
    # try:    
    #     axs[index].leg.set_visible(False)
    #     axs[0].leg.set_visible(True)
    # except AttributeError:
    #     pass
    plt.draw()

    ###### USE BLITTING FOR THUMBNAILS ######

# cid = fig.canvas.mpl_connect('button_press_event', onclick)
def on_key(event):
    try:
        axis = int(event.key)
        if axis == 1:
            pass
    except ValueError:
        return

    changeColor(axs[axis - 1])
    print axes_selected
    # switchAxes(axis - 1)


cid = fig.canvas.mpl_connect('key_press_event', on_key)

# reader = csv.reader(open('datatest.csv', 'rU'))
# reader1 = csv.reader(open('datatest1.csv', 'rU'))
# # reader2 = csv.reader(open('datatest2.csv', 'rU'))
# # reader3 = csv.reader(open('datatest3.csv', 'rU'))


axes_selected = None

for sensor in sensors:
    sensor.connect()

r0_bool = False
time0 = time.time()
basetime = 1
count = 0
# sensors[0].blitting = False
# for line in sensors[0].lines.values():
#     line.animated = False
# for row, row1 in zip(reader, reader1):
# for sensor in sensors:
    # if not sensors[0].stop:
        # try:
        #     sensors[0].update_data(row)
        #     sensors[1].update_data(row1)
            # sensors[2].update_data(row2)
            # sensors[3].update_data(row3)
            # count += 1
            # if count > 90:
            # plt.draw()
                # count = 0

        # except KeyboardInterrupt:
        #     print '\n'
        #     break


# HOSTS = ['sense03.local']#, 'sense03.local']
HOSTS = ['192.168.0.222']
# tns = [telnetlib.Telnet(HOST) for HOST in HOSTS]
writers = [csv.writer(open('recovery%d.csv' % (i), 'w')) for i in xrange(len(HOSTS))]

# HOST = 'sense02.local'
# tn = telnetlib.Telnet(HOST)
# writer = csv.writer(open('recovery.csv', 'w'))
# writer1 = csv.writer(open('recovery1.csv', 'w'))

var = '?'

init_count = 0

count = 0
dataz = []

press_events = set(["Lid Closed\r\n", "Sample Placed\r\n"])
exes = [['telnet', HOST] for HOST in HOSTS]
ps = [subprocess.Popen(exes[i], stdout=subprocess.PIPE, stderr=subprocess.STDOUT) for i in xrange(len(HOSTS))]
time.sleep(15)
while True:
# for i in xrange(len(HOSTS)):
    try:
        i = 0
        p = ps[i]
        writer = writers[i]
        retcode = p.poll() #returns None while subprocess is running
        data = p.stdout.readline()
        if retcode is not None:
            break
        if init_count < 4:
            init_count += 1 
            continue 
        # print data
        if not sensors[0].stop and init_count >= 3:
            count += 1
            if data == "Lid Closed\r\n":
                print '#########################'
                print 'HERE'
                print '~~~~~~~~~~~~~~~~~~~~~~~~~'
                count == 100
            print data
            if data[:4] == 'Temp':
                continue
            dataz.append(data)
            if count == 100:
                count = 0
                try:
                    # for i in xrange(len(HOSTS)):
                    sensors[i].update_data(data)
                    writer.writerow(data)
                except KeyboardInterrupt:
                    print '\n'
                    break
                    p.terminate()
    except KeyboardInterrupt:
        print '\n'
        break
        print dataz
        p.terminate()

# while True:
#     if not sensors[0].stop:
#         try:
#             for i in xrange(len(HOSTS)):
#                 tn = tns[i]
#                 sensor = sensors[i]
#                 tn.write(var)
#                 tn_read = str(tn.read_until(':\r\n'))
#                 sensor.update_data(tn_read)

#                 writer = writers[i]
#                 writer.writerow(tn_read)
#                 time.sleep(.2)

print dataz
data = sensor.export_data()
final_sensor = SensorFinal('test', data)
final_sensor.connect()


plt.show()