#!/usr/bin/env python
# pylint: disable=unused-import, unused-wildcard-import, no-self-argument, wildcard-import
# pylint: disable=too-many-instance-attributes, invalid-name

"""
Dynamic Sensor Plot
-----------------
Recieves live data from Arduino sensor, and plots real-time.
"""

from settings import *

class SensorLive(object):
    """
    Sensor class to graph live data
    """
    def __init__(self, name, fig=None, ax=None):
        self.fig = plt.figure() if fig is None else fig
        self.ax = self.fig.add_subplot(111) if ax is None else ax
        # self.ax.set_title('Sensor Resistance v. Time', fontsize=28)
        # self.ax.set_ylabel('$ng/mm^2$', fontsize=20)
        # self.ax.set_xlabel('Time [ms]', fontsize=20)
        # self.ax.tick_params(axis='both', labelsize=15)
        self.x_lim = [0, 1]
        self.y_lim_raw = [0, 5]
        self.y_lim_adjusted = [0, 5]
        self.name = name
        self.lines = {i:self.ax.plot(deque([]), deque([]), color=colors[i], linewidth=2.0, animated=False,
                                     label="ch"+str(i))[0] for i in xrange(9)}
        self.dummy_line = self.ax.plot([0][0], visible=False, scalex=False, scaley=False)
        self.ax.lines.pop()
        self.leg = self.legend_init()
        self.canvas = self.ax.figure.canvas
        self.y_data = [deque([]) for _ in xrange(9)]
        self.y_data_full = [[] for _ in xrange(9)]
        self.y_data_raw = [[] for _ in xrange(9)]
        self.x_data = [deque([]) for _ in xrange(9)]
        self.x_data_full = [[] for _ in xrange(9)]
        self.all_times = []
        self.visible = [True for _ in xrange(9)]
        self.live = [True for _ in xrange(9)]
        self.stop = False
        self.agent = None
        self.threshold = -0.01
        self.init_mult = 2.41
        self.lethal = 5
        self.maxline = self.ax.lines[0]
        self.maxy = 0
        self.ind = 0
        self.r_0 = [0.3 for _ in xrange(9)]
        self.ylim_max = 0
        self.vals = [{} for _ in xrange(9)]
        self.vals2 = [{} for _ in xrange(9)]
        self.count = [0 for _ in xrange(9)]
        self.channel_count = 0
        self.basetime = True
        self.speed = 1
        self.adjusted = False
        self.r0_step = 0
        self.init_mult = 2.41
        self.cidclick = None
        self.cidkey = None
        self.blitting = False
        self.lid_closed = None
        self.background = self.fig.canvas.copy_from_bbox(ax.bbox)
        plt.show(block=False)


    def blit(self): #, line):
        """
        Given a line, blits it on its respective graph.
        Args:
            channel (int): the channel to be updated, the index of self.lines
        """
        for line in self.lines.values():
            self.fig.canvas.restore_region(self.background)
            # for line in self.lines.values():
            self.ax.draw_artist(line)
            self.fig.canvas.blit(self.ax.bbox)

    def _run_once(some_func):
        def wrapper(*args, **kwargs):
            """ Makes sure a function in a loop runs once """
            # pylint: disable=not-callable
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

        leg = self.ax.legend(ncol=2, fancybox=True, shadow=True, loc='upper left',
                             framealpha=0.5, prop=font)
        leg.get_frame().set_alpha(0.4)
        leg.get_frame().set_facecolor('LightGreen')

        lined = {}
        for legline, origline in zip(leg.get_lines(), self.lines):
            legline.set_picker(7)  # 7 pts tolerance
            lined[legline] = origline

        return leg

    def connect(self):
        """
        Connects canvas of figure to selection events.
        """
        self.cidclick = self.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.cidkey = self.fig.canvas.mpl_connect('key_press_event', self.stop_func)

    def update_maxline(self):
        """
        Given a channel, identifies if maxline changes.
        Args:
            channel (int): the channel to be updated, the index of self.lines
        """
        for channel in xrange(9):
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

    @_run_once
    def plot_agent_indicator(self):
        """
        Plots vertical line marcating agent exposure.
        """
        self.ax.axvline(x=self.agent, color='b', linestyle='--')

    @_run_once
    def plot_lid_indicator(self):
        """
        Plots vertical line marcating lid being closed.
        """
        self.ax.axvline(x=self.lid_closed, color='r', linestyle='--')

    def onpick(self, event):
        """
        On a PickEvent, toggle lines/respective legend lines in the plot.
        Args:
            event (PickEvent): the artist of selected item of the plot
        """
        legline = event.artist
        channel = int(str(legline)[-2])
        self.visible[channel] = not self.visible[channel]

        vis = self.visible[channel]
        self.lines[channel].set_visible(vis)
        alpha = 1.0 if vis else 0.2
        legline.set_alpha(alpha)

    def stop_func(self, event):
        """
        On a KeyPressEvent, plot vertical line, or stop the live plot.
        Args:
            point (float): the raw resistance value to be updated
            channel (int): the channel to be updated, the index of self.lines
        """
        # if event.key == ' ':         # generates vertical line at spacebar press
        #     self.agent = self.all_times[-1] if self.agent is None else self.agent
        if event.key == 'ctrl+z':    # stops live graph generation
            self.stop = True

    def fill_line(self, interval, channel):
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
        startx, endx = interval[0], interval[-1]

        startval, endval = self.vals[channel][startx], self.vals[channel][endx]
        slope = float(endval - startval) / (endx - startx)

        filled_line = [0 for _ in xrange(9)]
        filled_line[0] = startval
        for i in xrange(1, 9):
            delta_x = interval[i] - interval[i - 1]
            filled_line[i] = filled_line[i - 1] + slope * delta_x

        return filled_line

    def live_off(self):
        # pylint: disable=no-self-use
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
        complete_data = [[] for _ in xrange(9)]
        for channel in xrange(9):
            index = channel
            for _ in xrange(channel):
                complete_data[channel].append(0)
            for _ in xrange((len(self.all_times))):
                try:
                    interval = self.all_times[index:index+10]
                    if len(interval) != 10:
                        break
                    filled_line = self.fill_line(interval, channel)
                    complete_data[channel].extend(filled_line)
                    index += 9
                except KeyError:
                    break
        # for i in xrange(9):
        #     for j in xrange(i):
        #         complete_data[i].insert(0,0)

        plot_data = self.all_times, complete_data
        visibility = self.live, self.visible
        etc = self.agent, self.ylim_max
        return plot_data, visibility, etc

    def adjust_data_point(self, resistance, channel):
        # pylint: disable=no-member
        """
        Given a resistance and channel, adjusts value to account for r_0.
        Args:
            resistance (float): the raw resistance value to be updated
            channel (int): the channel to be updated, the index of self.lines
        Returns:
            float : value adjusted for r_0 value
        """
        resistance = self.init_mult * np.log(float(resistance)/
                                             self.r_0[channel]) if resistance > 0 else 0
        self.live[channel] = False if resistance < self.threshold else True
        return resistance

    def fix_line(self, channel):
        """
        Given a channel, adjusts past data to account for changed r_0 value.
        Args:
            channel (int): the channel to be updated, the index of self.lines
        """
        for i in xrange(len(self.x_data[channel])):
            timestamp = self.x_data[channel][i]
            point = self.adjust_data_point(self.y_data_raw[channel][i], channel)
            self.y_data[channel][i] = point
            self.y_data_full[channel][i] = point
            self.vals[channel][timestamp] = point

    def update_data(self, raw_data):
        # pylint: disable=E1101
        """
        Given a string of raw data, updates database, line visibility, and plot.
        Args:
            raw_data (string): data read from the shell output
        Returns:
            float : value adjusted for r_0 value
        """
        channel, resistance, timestamp = self.parse_data(raw_data)

        if (channel, resistance, timestamp) == (0, 0, 0):
            return

        if resistance != 0 and self.r_0[channel] != 0.3:
            self.ylim_max = max(self.ylim_max, self.init_mult *
                                np.log(resistance/self.r_0[channel]))

        self.y_data_raw[channel].append(resistance)
        self.vals2[channel][timestamp] = resistance
        resistance = self.adjust_data_point(resistance, channel) if self.adjusted else resistance
        self.y_data[channel].append(resistance)
        self.x_data[channel].append(timestamp)
        self.y_data_full[channel].append(resistance)
        self.x_data_full[channel].append(timestamp)
        self.all_times.append(timestamp)
        self.vals[channel][timestamp] = resistance
        self.channel_count += 1 if not self.adjusted else self.speed

        xdata = self.x_data[channel]
        ydata = self.y_data[channel]

        if len(xdata) > 75:
            for _ in xrange(15):
                xdata.popleft()
                ydata.popleft()
            self.x_data[channel] = xdata
            self.y_data[channel] = ydata

        self.lines[channel].set_data(xdata, ydata)
        self.update_live_lines(channel, self.adjusted)

        if self.channel_count >= 9:    # Sporadic plotting to avoid mem overflow
            self.r0_step += 1
            self.maxy = 0

            if self.r0_step >= 10 and self.r_0[channel] == 0.3: # determines r0 for all channels
                for i in xrange(9):
                    if self.y_data[i][-1] != 0:
                        self.r_0[i] = self.y_data_raw[i][-1]
                        self.fix_line(i)
                        self.lines[i].set_ydata(self.y_data[i])

            self.channel_count = 0
            self.update_maxline()
            if not self.blitting:
                self.canvas.draw()
                self.canvas.flush_events()
            time.sleep(0.001)

            if self.blitting:
                self.blit()

    def parse_data(self, raw_data):
        """
        Given a resistance and channel, adjusts value to account for r_0.
        Args:
            resistance (float): the raw resistance value to be updated
            channel (int): the channel to be updated, the index of self.lines
        Returns:
            int : the channel to be updated, the index of self.lines
            float : value adjusted for r_0 value
            int : timestamp of measured resistance
        """
        if raw_data == "Lid Closed":
            self.lid_closed = self.all_times[-1] if self.lid_closed is None else self.lid_closed
            return 0, 0, 0

        if raw_data == "Sample Placed":
            self.agent = self.all_times[-1] if self.agent is None else self.agent
            return 0, 0, 0

        raw_data = raw_data.split(' , ')              # Uncomment these lines for live data source
        resistance = float(raw_data[3][:-2])          # That means this one, too
        channel, curtime = int(raw_data[0]), int(raw_data[2])
        # resistance = float(raw_data[2].split(' ')[0])   # Comment out for live data source

        if self.basetime is True:
            self.basetime = curtime
        curtime -= self.basetime

        if self.r_0[channel] != 0.3:
            self.adjusted = True

        return channel, resistance, curtime

    def update_live_lines(self, channel, adjusted):
        """
        Given a channel, updates the current maxline.
        Args:
            channel (int): the channel to be updated, the index of self.lines
            adjusted (bool): self.adjusted, denoting if r_0 has been defined
        """
        # update_leg = False
        ydata = self.y_data[channel]

        if ydata[-1] > self.threshold:
            update_lims = True
            if not self.live[channel]:
                # update_leg = True
                self.live[channel] = True
                self.ax.lines[channel] = self.lines[channel]
        else:
            update_lims = False
            if self.live[channel]:
                # update_leg = True
                self.live[channel] = False
                self.ax.lines[channel] = self.dummy_line

        # if update_leg:
        #     self.update_legend()

        if update_lims:
            self.update_lims(channel, adjusted)

        if self.lid_closed is not None:
            self.plot_lid_indicator()

        if self.agent is not None:
            self.plot_agent_indicator()

    def update_lims(self, channel, adjusted):
        """
        Given a channel, updates lims based on the channel's data.
        Args:
            channel (int): the channel to be updated, the index of self.lines
            adjusted (bool): self.adjusted, denoting if r_0 has been defined
        """
        y_max = 1
        for line in self.live:
            try:
                y_max = max(y_max, self.y_data[line][-1])
            except IndexError:
                pass

        if adjusted:
            self.y_lim_adjusted = [min(self.y_lim_adjusted[0], self.y_data[channel][-1] - 1),
                                   max(self.y_lim_adjusted[1], y_max + 3)]
            self.ax.set_ylim(self.y_lim_adjusted)
        else:
            self.y_lim_raw = [min(self.y_lim_raw[0], self.y_data[channel][-1] - 1), y_max + 3]
            self.ax.set_ylim(self.y_lim_raw)

        self.x_lim = [0, self.x_data[channel][-1] + 100000] # will depend on x-axis label
        self.ax.set_xlim(self.x_lim)

    def update_legend(self):
        """
        Regenerates legend.
        """
        self.leg.remove()
        self.leg = self.ax.legend(fancybox=True, shadow=True, loc='upper left',
                                  framealpha=0.5, prop=font)
        self.leg.get_frame().set_alpha(0.4)
        self.leg.get_frame().set_facecolor('LightGreen')

        for legline in self.leg.get_lines():
            legline.set_picker(7)  # 7 pts tolerance
