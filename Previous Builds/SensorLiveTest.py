#!/usr/bin/env python

"""
Static Sensor Plot
-----------------
Mimics live data from Arduino sensor, and plots real-time.
"""

from math import log
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# from matplotlib.widgets import  Button, CheckButtons, Cursor,\
#                                 RectangleSelector, Slider, SpanSelector

from drawnow import drawnow
import csv
import numpy as np
import time
import itertools
from datetime import datetime
import Tkinter
import FileDialog
import datetime
from SensorFinal import SensorFinal

basetime = True
# channel_count, r0_step = 0, 0
r0_step = 0
colors = ['b','g','r','c','m','y','k','purple','darkgreen']
font = FontProperties()
font.set_family('serif')
matplotlib.rc('font', family='serif')

threshold = -.01
init_mult = 2.41
lethal = 5

class SensorLive(object):
    """Sensor class to graph live data."""
    def __init__(self, name):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Sensor Resistance v. Time', fontsize=28)
        self.ax.set_ylabel('$ng/mm^2$', fontsize=20)
        self.ax.set_xlabel('Time [ms]', fontsize=20)
        self.ax.tick_params(axis='both', labelsize=15)
        self.x_lim = [0, 10]
        self.y_lim_raw = [0, 5]
        self.y_lim_adjusted = [0,5]
        self.name = name
        self.lines = {i:self.ax.plot([0], [0], color=colors[i], linewidth=2.0,
                                     label="ch"+str(i))[0] for i in xrange(9)}
        self.dummy_line = self.ax.plot([0][0], visible=False, scalex=False, scaley=False)
        self.leg = self.legend_init()
        self.canvas = self.ax.figure.canvas
        self.y_data = [[0] for _ in xrange(9)]
        self.y_data_complete = [[0 for _ in xrange(9)] for _ in xrange(9)]
        self.y_data_full = [[0] for _ in xrange(9)]
        self.y_data_raw = [[0] for _ in xrange(9)]
        self.x_data = [[0] for _ in xrange(9)]
        self.x_data_full = [[0] for _ in xrange(9)]
        self.all_times = []
        self.visible = [True for _ in xrange(9)]
        self.live = [True for _ in xrange(9)]
        self.stop = False
        self.agent = None
        self.threshold = -.01
        self.init_mult = 2.41
        self.lethal = 5
        self.maxline = self.ax.lines[0]
        self.maxy = self.maxline.get_ydata()[-1]
        self.ind = 0
        self.r_0 = [0.3 for _ in xrange(9)]
        self.ylim_max = 0
        self.vals = [{} for _ in xrange(9)]
        self.count = [0 for _ in xrange(9)]
        self.channel_count = 0
        self.basetime = 1
        plt.show(block=False)

    def _run_once(some_func):
        def wrapper(*args, **kwargs):
            """ Makes sure a function in a loop runs once """
            if not wrapper.has_run:
                wrapper.has_run = True
                return some_func(*args, **kwargs)
        wrapper.has_run = False
        return wrapper


    def legend_init(self):
        """
        Initializes the legend, establishes connection between legend lines
        and respective plot lines.
        Returns:
            LegendItem : legend of the graph
            dict (2DLine object): legend lines to plot lines
        """

        leg = self.ax.legend(fancybox=True, shadow=True, loc='upper left',
                             framealpha=0.5, prop=font)
        leg.get_frame().set_alpha(0.4)
        leg.get_frame().set_facecolor('LightGreen')

        lined = {}
        for legline, origline in zip(leg.get_lines(), self.lines):
            legline.set_picker(7)  # 7 pts tolerance
            lined[legline] = origline

        return leg

    def fix_lines(self):
        """
        Pop off the first zero values of all channels
        """
        pass

    def update_axes(self):
        self.canvas.draw()

    def remove_line(self, channel):
        pass

    def connect(self):
        """
        Connects canvas of figure to selection events.
        """
        self.cidclick = self.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.cidkey = self.fig.canvas.mpl_connect('key_press_event', self.stop_func)

    def update_figure_channel(self, channel):
        """
        Updates the data of a given channel, but does not refresh the graph.
        Args:
            channel (int): the channel to be updated, the index of self.lines
        """
        xdata = self.x_data[channel]
        ydata = self.y_data[channel]


        if self.agent is not None:
            self.plot_agent_indicator()

        self.lines[channel].set_data(xdata, ydata)
        self.x_lim = [0, max(self.x_lim[1], xdata[-1] + 100000)]
        self.y_lim_raw = [min(self.y_lim_raw[0], ydata[-1] - 1), max(self.y_lim_raw[1], ydata[-1] + 3)]
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim_raw)

    def update_figure(self):
        """
        Updates every channel on the graph.
        """
        for i in xrange(9):
            self.update_figure_channel(i)
        # axp = plt.axes([0.75,.89,.13,.13])
        # axp.axis('off')
        # timestamp = 'Local time: ' + str(datetime.time(datetime.now()))[:-5]
        # axp.text(0.5, 0.5, timestamp,weight='bold',fontsize=15,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    def adjust_data_channel(self, channel):
        """
        Given two points, calculates the euclidean distance (L2 norm) between them.
        Args:
            channel (int): the channel to be updated, the index of self.lines
        Returns:
            float list : resistance data updated with changed r_0 value
        """
        ydata = self.y_data[channel]
        new_ydata = []
        for point in ydata:
            new_ydata.append(0.3 * np.exp(point/self.init_mult))
            # new_ydata.append(meta)
        return [self.init_mult * np.log(float(y)/self.r_0[channel]) if y > 0 else y for y in new_ydata]

    def adjust_data_point(self, point, channel):
        """
        Given a resistance and channel, adjusts value to account for r_0.
        Args:
            point (float): the raw resistance value to be updated
            channel (int): the channel to be updated, the index of self.lines
        Returns:
            float : value adjusted for r_0 value
        """
        point = self.init_mult*np.log(float(point)/self.r_0[channel]) if point > 0 else 0 
        self.live[channel] = False if point < self.threshold else True
        return point

    def update_maxline(self, channel):
        """
        Given a channel, identifies if maxline changes.
        Args:
            channel (int): the channel to be updated, the index of self.lines
        """
        if self.visible[channel]:  # identify line with greatest current value
            temp = self.maxy
            self.maxy = max(self.maxy, self.y_data[channel][-1])
            self.maxline = self.lines[channel] if temp != self.maxy else self.maxline
            self.ind = channel if temp != self.maxy else self.ind

            if self.maxline.get_ydata() != 0:    # fill from maxline and the x axis
                lethal = np.array([self.lethal for _ in self.maxline.get_ydata()])
                maxline = np.array(self.maxline.get_ydata())
                for _ in self.ax.collections:    # delete previous fill_between()s
                    self.ax.collections.pop(-1)
                self.ax.fill_between(self.x_data[self.ind], maxline, facecolor='green',
                                     alpha=0.3, where=maxline > lethal)

        # self.ax.draw_artist(filler)
        # self.canvas.blit(self.ax.bbox)
        # self.canvas.draw()
        # time.sleep(0.01)
        # self.canvas.flush_events()
        # plt.pause(0.0001)

    @_run_once
    def plot_agent_indicator(self):
        """
        Plots vertical line marcating agent exposure.
        """
        self.ax.axvline(x=self.agent, color='b', linestyle='--')

    def update_data(self, channel, resistance, timestamp):
        """
        Updates channel data with timestamp and resistance.
        Args:
            resistance (float): the raw resistance value to be updated
            channel (int): the channel to be updated, the index of self.lines
            timestamp (int): time of resistance recording
        """

        self.y_data_raw[channel].append(resistance)
        resistance = self.adjust_data_point(resistance, channel)
        self.y_data[channel].append(resistance)
        self.x_data[channel].append(timestamp)
        self.y_data_full[channel].append(resistance)
        self.x_data_full[channel].append(timestamp)
        self.all_times.append(timestamp)
        self.vals[channel][timestamp] = resistance
        self.count[index] += 1
        self.channel_count += 1

        if self.count[channel] >= 2:
            interval = self.all_times[-10:]
            new_line = self.fill_line(self.vals, interval, channel)
            self.y_data_complete[channel].extend(new_line)

        # # self.ax.set_xlim([0, timestamp])
        # # self.ax.set_ylim([0, self.ylim_max])

    def onpick(self, event):
        """
        On a PickEvent, toggle lines, respective legend lines in the plot.
        Args:
            event (PickEvent): the artist of selected item of the plot
        """
        legline = event.artist
        ch = int(str(legline)[-2])
        self.visible[ch] = not self.visible[ch]

        vis = self.visible[ch]
        self.lines[ch].set_visible(vis)
        alpha = 1.0 if vis else 0.2
        legline.set_alpha(alpha)

    def stop_func(self, event):
        """
        On a KeyPressEvent, plot vertical line, or stop the live plot.
        Args:
            point (float): the raw resistance value to be updated
            channel (int): the channel to be updated, the index of self.lines
        """
        if event.key == ' ':     # generates vertical line at spacebar press
            self.agent = self.all_times[-1] if self.agent is None else self.agent
        if event.key == 'ctrl+z': # stops live graph generation
            self.stop = True

    def fill_line(self, vals, interval, channel):
        """
        Backfill resistance values at every timestep in every channel. Necessary
        for smooth fill_between() on final graph.
        Args:
            vals (dict float): dictionary of values per channel
            interval (list int): the time interval to be filled
            channel (int): the channel to be updated, the index of self.lines 
        Returns:
            list (float): completed interval of resistances in channel
        """
        startx, endx = interval[0],interval[-1]

        startval, endval = vals[channel][startx], vals[channel][endx]
        slope = float(endval - startval) / (endx - startx)

        filled_line = [0 for _ in xrange(9)]
        filled_line[0] = startval
        for i in xrange(1, 9):
            delta_x = interval[i] - interval[i - 1]
            filled_line[i] = filled_line[i - 1] + slope * delta_x

        return filled_line

    def live_off(self):
        """
        Keeps the current figure showing after the static graphs are generated.
        """
        plt.show(block=True)

    def export_data(self):
        """
        Backfill resistance values at every timestep in every channel. Necessary
        for smooth fill_between() on final graph.
        Returns:
            list (float): completed interval of resistances in channel
            dict (bool): lines currently active in the plot
            dict (bool): lines currently visible in the plot
            list (Line2D Object): list of current lines 
        """
        plot_data = self.all_times, self.y_data_complete
        visibility = self.live, self.visible
        etc = self.agent, self.ylim_max#self.ax.get_ylim()[1]
        return plot_data, visibility, etc

    def f(self, point, channel):
        if self.r_0[channel] == 0:
            self.live[channel] = False
            return 0
        return self.init_mult * np.log(point/self.r_0[channel])

    def fix_line(self, channel):
        self.y_data_complete[channel] = []
        start, end = -10, None
        for i in xrange(len(self.x_data[channel])):
            timestamp = self.x_data[channel][i]
            point = self.f(self.y_data_raw[channel][i], channel)
            self.y_data[channel][i] = point
            self.y_data_full[channel][i] = point
            self.vals[channel][timestamp] = point
            # if i >= 1:
            #     end = self.all_times.index(timestamp)
            #     start = end - 10
            #     print timestamp
            #     print end
            #     interval = self.all_times[start:end]
            #     print interval
            #     new_line = self.fill_line(self.vals, interval, channel)
            #     self.y_data_complete[channel].extend(new_line)














    def update_raw(self, channel, resistance, timestamp):
        self.y_data_raw[channel].append(resistance)
        # resistance = self.adjust_data_point(resistance, channel)
        self.y_data[channel].append(resistance)
        self.x_data[channel].append(timestamp)
        self.y_data_full[channel].append(resistance)
        self.x_data_full[channel].append(timestamp)
        self.all_times.append(timestamp)
        self.vals[channel][timestamp] = resistance
        self.count[index] += 1
        self.channel_count += 1

        # if self.count[channel] >= 2:
        #     interval = self.all_times[-10:]
        #     new_line = self.fill_line(self.vals, interval, channel)
        #     self.y_data_complete[channel].extend(new_line)


        xdata = self.x_data[channel]
        ydata = self.y_data[channel]

        self.lines[channel].set_data(xdata, ydata)

        if ydata[-1] > 0: #self.threshold:
            update_lims = True
            if not self.live[channel]:
                self.live[channel] = True
                self.ax.lines[channel] = self.lines[channel]
        else:
            update_lims = False
            if self.live[channel]:
                self.live[channel] = False
                self.ax.lines[channel] = self.dummy_line

        self.leg.remove()

        leg = self.ax.legend(fancybox=True, shadow=True, loc='upper left',
                             framealpha=0.5, prop=font)
        leg.get_frame().set_alpha(0.4)
        leg.get_frame().set_facecolor('LightGreen')

        for legline in leg.get_lines():
            legline.set_picker(7)  # 7 pts tolerance

        if self.agent is not None:
            self.plot_agent_indicator()

        if update_lims:
            y_max = 1
            for channel in self.live:
                y_max = max(y_max, self.y_data[channel][-1])
            self.x_lim = [0, xdata[-1] + 100000]
            self.y_lim_raw = [min(self.y_lim_raw[0], ydata[-1] - 1), y_max + 3]# max(self.y_lim_raw[1], ydata[-1] + 3)]
            self.ax.set_xlim(self.x_lim)
            self.ax.set_ylim(self.y_lim_raw)




# vals = {i:{0:0} for i in xrange(9)}
# all_times = []
# count = [0 for i in xrange(9)]

reader = csv.reader(open('datatest.csv', 'rU'))
sensors = [SensorLive('sense0'+ str(i)) for i in xrange(1,2)]

# for i in xrange(len(sensors)):
#     sensor = sensors[i]
count = 0
old_time = 0
sensor = sensors[0]
sensor.connect()
r0_bool = False
for row in reader:
    if not sensor.stop:
        try:
            # sensor.channel_count += 1
            index, curtime = int(row[0]), int(row[1])
            resistance = float(row[2].split(' ')[0])
            basetime = curtime if basetime else basetime
            curtime -= basetime
            sensor.update_data(index, resistance, curtime)
            sensor.update_figure_channel(index)
            print "Channel Count: ", sensor.channel_count

            if resistance != 0 and sensor.r_0[index] != 0.3:
                print "%d should not be 0.3 " %(sensor.r_0[index])
                sensor.ylim_max = max(sensor.ylim_max, init_mult * np.log(resistance/sensor.r_0[index]))
                print sensor.ylim_max

            if sensor.channel_count >= 9:    # Needed to simulate live datastream
                r0_step += 1

                if r0_step >= 10 and sensor.r_0[index] == 0.3: # determines r0 for all channels
                    for i in xrange(9):
                        # i = index
                        if sensor.y_data[i][-1] != 0:
                            print '##########################'
                            print 'UPDATED'
                            # sensor.r_0[i] = sensor.y_data[i][-1]
                            sensor.r_0[i] = sensor.y_data_raw[i][-1]
                            sensor.fix_line(i)
                            sensor.lines[i].set_ydata(sensor.y_data[i])
                            # new_data = sensor.adjust_data_channel(i)
                            # sensor.lines[i].set_ydata(new_data)
                            # sensor.y_data[i] = new_data
                            r0_bool = True

                # if not r0_bool: 
                sensor.channel_count = 0
                sensor.canvas.draw()
                sensor.canvas.flush_events()
                plt.pause(.0001)

            # sensor.update_maxline(index)
            # sensor.update_figure_channel(index)
            print "r0_step: ", r0_step

        except KeyboardInterrupt:
            print '\n'
            break

data = sensor.export_data()
final_sensor = SensorFinal('test', data)
final_sensor.connect()
