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
from matplotlib.widgets import  Button, CheckButtons, Cursor,\
                                RectangleSelector, Slider, SpanSelector

from drawnow import drawnow
import csv
import numpy as np
import time
import itertools
from datetime import datetime
import Tkinter
import FileDialog
import datetime

basetime = True
channel_count, r0_step = 0, 0
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
        # self.fig, self.ax = plt.subplots() # figsize=(24, 18))
        # self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.set_title('Sensor Resistance v. Time', fontsize=28)
        self.ax.set_ylabel('$ng/mm^2$', fontsize=20)
        self.ax.set_xlabel('Time [ms]', fontsize=20)
        self.ax.tick_params(axis='both', labelsize=15)
        self.x_lim = [0, 10]
        self.y_lim = [0, 5]
        self.name = name
        self.lines = [self.ax.plot([0], [0], color=colors[i], linewidth=2.0,
                      label="ch"+str(i))[0] for i in xrange(9)]
        self.canvas = self.ax.figure.canvas
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.leg, self.lined = self.legend_init()
        self.y_data = [[0] for _ in xrange(9)]
        self.y_data_complete = [[0 for _ in xrange(i)] for i in xrange(9)]
        self.y_data_full = [[0] for _ in xrange(9)]
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
        self.r_0 = [0.3 for i in xrange(9)]
        self.ylim_max = 0
        self.vals = [{0:0} for i in xrange(9)]
        self.count = [0 for i in xrange(9)]
        self.channel_count = 0
        plt.show(block=False)

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

        return leg, lined


    def update_axes(self):
        self.canvas.draw()

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
        self.x_lim = [0, max(self.x_lim[1], xdata[-1])]
        self.y_lim = [min(self.y_lim[0], ydata[0] - 1), max(self.y_lim[1], ydata[-1] + 3)]
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

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
        return [self.init_mult*log(float(y)/self.r_0[channel]) if y > 0 else y for y in ydata]

    def adjust_data_point(self, point, channel):
        """
        Given a resistance and channel, adjusts value to account for r_0.
        Args:
            point (float): the raw resistance value to be updated
            channel (int): the channel to be updated, the index of self.lines
        Returns:
            float : value adjusted for r_0 value
        """
        return self.init_mult*log(float(point)/self.r_0[channel]) if point > 0 else 0 

    def update_maxline(self, channel):
        """
        Given a channel, identifies if maxline changes, updates plot.
        Args:
            channel (int): the channel to be updated, the index of self.lines
        """
        if self.visible[channel]:  # identify line with greatest current value
            temp = self.maxy
            self.maxy = max(self.maxy, self.y_data[channel][-1])
            self.maxline = self.lines[channel] if temp != self.maxy else self.maxline
            self.ind = channel if temp != self.maxy else self.ind

        lethal_y = np.array(self.lethal for i in xrange(len(self.maxline.get_ydata())))
        if self.maxline.get_ydata() != 0:    # fill from maxline and the x axis
            y2 = np.array([lethal for i in self.maxline.get_ydata()])
            maxline = np.array(self.maxline.get_ydata())
            for _ in self.ax.collections:
                self.ax.collections.pop(-1)
            self.ax.fill_between(self.x_data[self.ind], maxline, facecolor='green',
                                 alpha=0.3, where=maxline > y2)

        # self.ax.draw_artist(filler)
        # self.canvas.blit(self.ax.bbox)
        self.canvas.draw()
        # time.sleep(0.01)
        self.canvas.flush_events()
        plt.pause(0.0001)

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
        resistance = self.adjust_data_point(resistance, channel)
        self.y_data[channel].append(resistance)
        self.x_data[channel].append(timestamp)
        self.y_data_full[channel].append(resistance)
        self.x_data_full[channel].append(timestamp)
        self.all_times.append(timestamp)
        self.vals[channel][timestamp] = resistance
        self.channel_count += 1
        self.count[index] += 1

        if self.count[channel] >= 2:
            interval = self.all_times[-10:]
            new_line = self.fill_line(self.vals, interval, channel)
            self.y_data_complete[channel].extend(new_line)

        # if resistance != 0 and self.r_0[channel] != 0.3:
        #     self.ylim_max = max(self.ylim_max, resistance + 5)

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

        for legline, origline in zip(self.leg.get_lines(), self.lines):  # toggle line visibility with legend
            legline.set_picker(7)  # 7 pts tolerance
            self.lined[legline] = origline
            ch = int(str(legline)[-2])
            vis = self.visible[ch]
            origline.set_visible(vis)
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
        etc = self.agent, self.ylim_max
        return plot_data, visibility, etc




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
for row in reader:
    if not sensor.stop:
        try:
            index, curtime = int(row[0]), int(row[1])
            resistance = float(row[2].split(' ')[0])
            basetime = curtime if basetime else basetime
            curtime -= basetime
            sensor.update_data(index, resistance, curtime)
            # sensor.channel_count += 1
            # sensor.count[index] += 1

            # if sensor.count[index] >= 2:
            #     interval = sensor.all_times[-10:]
            #     new_line = sensor.fill_line(sensor.vals, interval, index)
            #     sensor.y_data_complete[index].extend(new_line)

            if resistance != 0 and sensor.r_0[index] != 0.3:
                sensor.ylim_max = max(sensor.ylim_max, init_mult*log(resistance/sensor.r_0[index]))

            if channel_count == 9:    # Needed to simulate live datastream
                r0_step += 1
                if r0_step >= 10 and sensor.r_0[index] == 0.3: # determines r_0 for all y_data
                    for i in xrange(9):
                        if sensor.y_data[i][-1] != 0:
                            sensor.r_0[i] = sensor.y_data[i][-1]
                            new_data = sensor.adjust_data_channel(index)
                            sensor.lines[index].set_ydata(new_data)
                sensor.channel_count = 0
            sensor.update_figure_channel(index)
                # sensor.update_figure()
            sensor.update_maxline(index)

        except KeyboardInterrupt:
            print '\n'
            sensor.live_off()
            data = sensor.export_data()
            final_sensor = SensorFinal('test', data)
            break

# sensor.live_off()





class SensorFinal(object):
    def __init__(self, name, data):
    # def __init__(self, name):
        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111)
        # self.fig, self.ax = plt.subplots() # figsize=(24, 18))
        # self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        # self.ax.set_title('Sensor Resistance v. Time', fontsize=28)
        # self.ax.set_ylabel('$ng/mm^2$', fontsize=20)
        # self.ax.set_xlabel('Time [ms]', fontsize=20)
        # self.ax.tick_params(axis='both', labelsize=15)
        # self.x_lim = [0, 10]
        # self.y_lim = [0, 5]
        self.fig = plt.figure(dpi=100, facecolor='#9EC4A8') # figsize=(24, 18))
        self.ax = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.name = name
        # self.lines = [self.ax.plot([0], [0], color=colors[i], linewidth=2.0,
        #               label="ch"+str(i))[0] for i in xrange(9)]
        self.canvas = self.fig.canvas
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.lines, self.lines2 = self.plot_lines_init()
        self.leg, self.lined, self.lined2 = self.legend_init()
        self.span, self.rect, self.cursor, self.check, self.slider = self.widget_init()
        # self.x_data_full = [[0] for _ in xrange(9)]
        self.init_mult = 2.41
        self.lethal = 5

        #### ESTABLISH UNPACKING INIT FOR all_times AND y_data_full# 
        data_gen = self.unpack_init_values(self, data)
        self.all_times, self.y_data_complete = data_gen.next()
        self.live, self.visible = data_gen.next()
        self.agent, self.ylim_max = data_gen.next()
        #######################################

        self.span_name, self.rect_name = 'Span', 'Rect'
        self.threshold = -.01

        self.maxline = self.ax.lines[0]
        self.maxy = self.maxline.get_ydata()[-1]
        self.ind = 0
        # self.r_0 = [0.3 for i in xrange(9)]
        plt.show()
##############################################
        # self.fig = plt.figure(dpi=100,facecolor='#9EC4A8') # figsize=(24, 18))
        # self.ax = self.fig.add_subplot(211)
        # self.ax2 = self.fig.add_subplot(212)

        # lines, lines2 = {}, {}

    def unpack_init_values(self, data):
        """ doc string """

        # plot_data = self.all_times, self.y_data_complete
        # visibility = self.live, self.visible
        # etc = self.agent, self.ylim_max
        def gen():
            for dataset in data:
                yield dataset
        return gen()
        # return plot_data, visibility, etc



    def plot_lines_init(self):
        """
        Plot all live lines and returns lists of lines for both plots.
        Returns:
            list (Line2D object): plot line objects for upper graph
            list (Line2D object): plot line objects for lower graph
        """
        self.ax.set_title('Sensor Resistance v. Time', fontsize=28, fontproperties=font)
        self.ax.set_ylabel('$ng/mm^2$', fontsize=20)
        self.ax.set_xlabel('Time [ms]', fontsize=20)
        self.ax.set_ylim(-0.5, self.ylim_max + 1)
        self.ax.tick_params(axis='both', labelsize=15)

        self.ax2.set_ylabel('$ng/mm^2$', fontsize=20)
        self.ax2.set_xlabel('Time [ms]', fontsize=20)
        self.ax2.set_ylim(-0.5, self.ylim_max + 1)
        self.ax2.tick_params(axis='both', labelsize=15)

        lines, lines2 = {}, {}
        for i in (i for i in xrange(9) if self.live[i]):
            # y_data_full[i] = [init_mult*log(float(r)/r_0[i]) if r > 0 else 0 for r in y_data_complete[i]]

            if self.y_data_complete[i][-1] < 0: # if last value is negative, disregard the associated line
                self.live[i] = False
                continue

            r =  self.y_data_complete[i][-1] # last adjustments to final filled lines
            # self.y_data)complete[i].append(r)
            while len(self.y_data_complete[i]) != len(self.all_times):
                self.y_data_complete[i].append(r)

            line, = self.ax.plot(self.all_times, self.y_data_complete[i], color=colors[i],
                                 label='ch'+str(i), linewidth=2) # UPDATE EACH CHANNEL
            line2, = self.ax2.plot(self.all_times, self.y_data_complete[i], color=colors[i],
                                   label='ch'+str(i), linewidth=2)
            lines[i] = line
            lines2[i] = line2

        if self.agent != None: # vertical lines
            self.ax.axvline(x=self.agent, color='b', linestyle='--')
            self.ax2.axvline(x=self.agent, color='b', linestyle='--')

        return lines, lines2



        # put this in legend init #
    def legend_init(self):
        """
        Initilizes legend for both plots, links legend lines to plot lines
        on both plots.
        Returns:
            LegendItem : legend of the graph
            dict (2DLine object): legend lines to plot lines of upper plot
            dict (2DLine object): legend lines to plot lines of upper plot
        """
        leg = self.ax.legend(loc='upper left', fancybox=True, shadow=True)
        leg.get_frame().set_alpha(0.4)
        leg.get_frame().set_facecolor('#83B389') # greenish, kinda
        lined, lined2 = {}, {}

        for legline, origline, origline2 in zip(leg.get_lines(), self.lines.values(),
                                                self.lines2.values()):
            legline.set_picker(7)  # 7 pts tolerance
            lined[legline] = origline
            lined2[legline] = origline2
            ch = int(str(legline)[-2])
            vis = self.visible[ch]
            origline.set_visible(vis)
            origline2.set_visible(vis)
            alpha = 1.0 if vis else 0.2
            legline.set_alpha(alpha)

        ############################

        self.fill_below()
        return leg, lined, lined2

    def widget_init(self):
        """ docstring """
        span = SpanSelector(self.ax, self.onselect, 'horizontal', useblit=True, minspan=5,
                            rectprops=dict(alpha=0.5, facecolor='red'))
        rect = RectangleSelector(self.ax, self.onselect_rect, drawtype='box', useblit=True,
                                 rectprops=dict(alpha=0.5, facecolor='red'))
        cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)

        cursor.active, span.active, rect.active = False, False, False
        plt.subplots_adjust(left=0.2, hspace=0.5)
        axc = plt.axes([0.05, .45, .1, .10]) # will have to adjust depending on span_name/rect_name
        axs = plt.axes([0.25, .48, .2, .03])
        # axp = plt.axes([0.025,.85,.13,.13])

        check = CheckButtons(axc, (self.span_name, self.rect_name), (False, False))
        check.on_clicked(self.func)

        nicotine = 2.41
        slider = Slider(axs, 'Multiplier', 0.1, 5, valinit=nicotine)
        slider.on_changed(self.update)
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())

        return span, rect, cursor, check, slider

    def connect(self):
        """ doc string """
        plt.connect('pick_event', self.onpick_f)
        plt.connect('button_press_event', self.cursor_off)
        plt.connect('button_release_event', self.cursor_on)

    def onpick_f(self, event):
        """ docstring"""
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        ch = int(str(legline)[-2])
        self.visible[ch] = not self.visible[ch]
        origline = self.lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)

        origline2 = self.lined2[legline] # for zoomed in region
        vis = not origline2.get_visible()
        origline2.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)

        self.fill_below()

    def fill_below(self):
        """
        Fills between maximum point at every timestep and x-axis
        where the value of the point > self.lethal.
        Returns:
            None :  if no lines are currently visible
        """
        available = (i for i in xrange(9) if self.live[i] * self.visible[i])
        maxlines = [self.lines2[j].get_ydata() for j in available]
        maxtimes = [self.lines2[j].get_xdata() for j in available]

        if len(maxlines) == 0: # in case no lines are currently visible
            if len(self.ax2.collections):
                self.ax2.collections.pop()
            if len(self.ax.collections):
                self.ax.collections.pop()
            self.fig.canvas.draw()
            return

        final_len = len(maxlines[-1]) 

        for linea in maxlines:
            if len(linea) != final_len:
                linea.pop(-1)

        yfill = [0 for i in xrange(final_len)]
        tfill = [0 for i in xrange(final_len)]
        llines, ltimes = list(zip(*maxlines)), list(zip(*maxtimes))

        for i in xrange(final_len):  
            yfill[i] = max(llines[i])
            tfill[i] = ltimes[i][llines[i].index(yfill[i])]

        y2 = np.array([lethal] * final_len)

        if len(self.ax2.collections): # avoid overwriting fill_betweens
            self.ax2.collections.pop()
        if len(self.ax.collections):
            self.ax.collections.pop()

        self.ax2.fill_between(tfill, np.array(yfill), facecolor='#F73C3C',
                              where=yfill > y2)#,alpha=0.3)#, interpolate=True)
        self.ax.fill_between(tfill, np.array(yfill), facecolor='#F73C3C',
                             where=yfill > y2)#,alpha=0.3)#, interpolate=True)
        self.fig.canvas.draw()


# # ####################### SPAN SELECTOR #######################
    def onselect(self, xmin, xmax):
        x = self.all_times
        indmin, indmax = np.searchsorted(x, (xmin, xmax))
        indmax = min(len(x) - 1, indmax) 
        thisy_min, thisy_max = float('inf'),-float('inf')

        for i in (i for i in xrange(9) if self.live[i]):
            data = self.lines2[i].get_ydata()
            thisy_min = min(thisy_min, data[indmin])
            thisy_max = max(thisy_max, data[indmax])

            # If data sets are huge, it may be necessary to slice
            # This could cause issues in general widget cohesiveness

            # xdata = list(itertools.islice(x_data_full[i],max(0,indmin-5),
            #                               min(indmax+5,len(x)-1)))
            # ydata = list(itertools.islice(y_data_full[i],max(0,indmin-5),
            #                               min(indmax+5,len(x)-1)))
            # lines2[i].set_data(xdata,ydata)

        self.ax2.set_xlim(xmin, xmax)
        self.ax2.set_ylim(thisy_min-2, thisy_max+2) # window cushion
        self.fig.canvas.draw()
# #############################################################

# ##################### RECTANGLE SELECTOR #################### 
    def onselect_rect(self, eclick, erelease):
        x_init, y_init = eclick.xdata, eclick.ydata
        x_final, y_final = erelease.xdata, erelease.ydata  

        # If data sets are huge, it may be necessary to slice
        # This could cause issues in general widget cohesiveness

        # xmin, xmax = min(x_init,x_final), max(x_init, x_final)
        # ymin, ymax = min(y_init,y_final), max(y_init, y_final)

        # x = x_data_full[0] # arbitrary selection from x_data_full
        # indmin, indmax = np.searchsorted(x, (xmin, xmax))
        # indmax = min(len(x) - 1, indmax)

        # for i in xrange(9):
        #     if not live[i]: # blacked out lines
        #         continue
        #     xdata = list(itertools.islice(x_data_full[i], max(0,indmin-5),
        #                                   min(indmax+5, len(x)-1)))
        #     ydata = list(itertools.islice(y_data_full[i], max(0,indmin-5),
        #                                   min(indmax+5, len(x)-1)))
        #     lines2[i].set_data(xdata, ydata)

        self.ax2.set_xlim(x_init, x_final)
        self.ax2.set_ylim(y_init, y_final) # window cushion
        self.fig.canvas.draw()
# #############################################################

# ############## CHECKBUTTON FOR GRAPH SELECTOR ###############
    def func(self, label):
        if label == self.span_name:
            self.span.active = not self.span.active
        elif label == self.rect_name:
            self.rect.active = not self.rect.active
            self.cursor.active = self.rect.active
        self.fig.canvas.draw()
# #############################################################

# ##################### CURSOR VISIBILITY #####################
    def cursor_off(self, event):
        if self.rect.active:
            self.cursor.active = False

    def cursor_on(self, event):
        if self.rect.active:
            self.cursor.active = True
# #############################################################

# ##################### SLIDER FUNCTION #######################
    def update(self, val):
        multiplier = self.slider.val
        ylim_max = 0
        xlim = self.ax2.get_xlim()
        ylim2 = 0
        x = self.all_times
        indmin, indmax = np.searchsorted(x, xlim)
        indmax = min(len(x) - 1, indmax) 
        for i in (i for i in xrange(9) if self.live[i]):
            mult = [multiplier/init_mult*num for num in self.y_data_complete[i]]
            ylim_max = max(ylim_max, max(mult))
            ylim2 = max(ylim2,max(mult[:indmax]))
            self.lines[i].set_ydata(mult)
            self.lines2[i].set_ydata(mult)
        if self.agent != None:
            self.ax.axvline(x=self.agent, color='b', linestyle='--')
            self.ax2.axvline(x=self.agent, color='b', linestyle='--')

        self.ax.set_ylim(-0.5, ylim_max + 1)
        self.ax2.set_ylim(-0.5, ylim2 + 1)
        self.fill_below()
# #############################################################

# #################### STATIC FINAL GRAPH #####################
# fig = plt.figure(dpi=100,facecolor='#9EC4A8') # figsize=(24, 18))
# ax = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)

# lines, lines2 = {}, {}

# ax.set_title('Sensor Resistance v. Time',fontsize=28,fontproperties=font)
# ax.set_ylabel('$ng/mm^2$',fontsize=20)
# ax.set_xlabel('Time [ms]',fontsize=20)
# ax.set_ylim(-0.5,ylim_max+1)
# ax.tick_params(axis='both', labelsize=15)

# ax2.set_ylabel('$ng/mm^2$',fontsize=20)
# ax2.set_xlabel('Time [ms]',fontsize=20)
# ax2.set_ylim(-0.5,ylim_max+1)
# ax2.tick_params(axis='both', labelsize=15)

# for i in (i for i in xrange(9) if live[i]):
#     ch = strchannels[i]
#     y_data_full[i] = [init_mult*log(float(r)/r_0[i]) if r > 0 else 0 for r in y_data_complete[i]]

#     if chval(i)[-1] < 0: # if last value is negative, disregard the associated line
#         live[i] = False
#         continue

#     r =  y_data[i][-1] # last adjustments to final filled lines
#     y_data_full[i].append(init_mult*log(float(r)/r_0[i]))
#     for j in xrange(8-i):
#         y_data_full[i].append(y_data_full[i][-1]) 
 
#     line, = ax.plot(all_times,y_data_full[i],color=colors[i],label=ch,linewidth=2) # UPDATE EACH CHANNEL
#     line2, = ax2.plot(all_times,y_data_full[i],color=colors[i],label=ch,linewidth=2)
#     lines[i] = line
#     lines2[i] =line2

# if agent != None: # vertical lines
#         ax.axvline(x=agent,color='b',linestyle='--')
#         ax2.axvline(x=agent,color='b',linestyle='--')

# leg = ax.legend(loc='upper left', fancybox=True, shadow=True)
# leg.get_frame().set_alpha(0.4)
# leg.get_frame().set_facecolor('#83B389') # greenish, kinda
# lined, lined2 = {}, {}

# for legline, origline, origline2 in zip(leg.get_lines(), lines.values(), lines2.values()):
#     legline.set_picker(7)  # 7 pts tolerance
#     lined[legline] = origline
#     lined2[legline] = origline2
#     ch = int(str(legline)[-2])
#     vis = visible[ch]
#     origline.set_visible(vis)
#     origline2.set_visible(vis)
#     alpha = 1.0 if vis else 0.2
#     legline.set_alpha(alpha)

# fill_below()

# ##########################################################

# span = SpanSelector(ax, onselect, 'horizontal', useblit=True, minspan=5,
#                     rectprops=dict(alpha=0.5, facecolor='red'))
# rect = RectangleSelector(ax, onselect_rect, drawtype='box',useblit=True,
#                     rectprops=dict(alpha=0.5, facecolor='red'))
# cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

# span_name, rect_name = 'Span', 'Rect'
# cursor.active, span.active, rect.active = False, False, False
# plt.subplots_adjust(left=0.2,hspace=0.5)
# axc = plt.axes([0.05,.45,.1,.10]) # will have to adjust depending on span_name/rect_name
# axs = plt.axes([0.25,.48,.2,.03])
# # axp = plt.axes([0.025,.85,.13,.13])

# check = CheckButtons(axc, (span_name, rect_name) , (False, False))
# check.on_clicked(func)

# nicotine = 2.41
# slider = Slider(axs, 'Multiplier', 0.1, 5, valinit=nicotine)
# slider.on_changed(update)

# plt.connect('pick_event', onpick_f)
# plt.connect('button_press_event', cursor_off)
# plt.connect('button_release_event', cursor_on)
# #############################################################
# manager = plt.get_current_fig_manager()
# manager.resize(*manager.window.maxsize())
    
# datafile = 'logo.png'
# img = imread(datafile)
# axp.imshow(img)
# axp.axis('off')

plt.show()