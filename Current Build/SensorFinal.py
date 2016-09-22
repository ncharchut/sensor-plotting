#!/usr/bin/env python
# pylint: disable=unused-import, unused-wildcard-import, no-self-argument, wildcard-import
# pylint: disable=too-many-instance-attributes, invalid-name, no-self-use, unused-argument

"""
Static Sensor Plot
-----------------
Takes in recorded data for final plot.
"""

from settings import *
from matplotlib.widgets import  Button, CheckButtons, Cursor,\
                                RectangleSelector, Slider, SpanSelector

class SensorFinal(object):
    """ Static final graph for sensor """
    def __init__(self, name, data):
        self.fig = plt.figure(dpi=100, facecolor='#9EC4A8') # figsize=(24, 18))
        self.ax = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.name = name
        self.lethal = 5
        self.init_mult = 2.41
        self.span_name, self.rect_name = 'Span', 'Rect'
        self.threshold = -.01
        self.canvas = self.fig.canvas
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        data_gen = self.unpack_init_values(data)
        self.all_times, self.y_data_complete = data_gen.next()
        self.live, self.visible = data_gen.next()
        self.agent, self.ylim_max = data_gen.next()
        self.lines, self.lines2 = self.plot_lines_init()
        self.leg, self.lined, self.lined2 = self.legend_init()
        self.span, self.rect, self.cursor,\
                   self.check, self.slider, self.button = self.widget_init()
        self.connect()
        plt.show()

    def unpack_init_values(self, data):
        """
        Unpack initial values exported from SensorLive object.
        Args:
            data (3x2 tuple list float): ((self.all_times, self.y_data_complete),
                                          (self.live, self.visible),
                                          (self.agent, self.ylim_max))
        Returns:
            generator : to unpack tuple
        """
        def gen():
            """
            Iterates through imported data and unpacks.
            """
            for dataset in data:
                yield dataset
        return gen()

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
            # y_data_full[i] = [init_mult*log(float(r)/r_0[i]) if r > 0 else 0
            #                   for r in y_data_complete[i]]

            if self.y_data_complete[i][-1] < 0: # if negative, disregard the associated line
                self.live[i] = False
                continue

            r = self.y_data_complete[i][-1] # last adjustments to final filled lines
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

    def legend_init(self):
        """
        Initilizes legend for both plots, links legend lines to plot lines
        on both plots.
        Returns:
            LegendItem : legend of the graph
            dict (2DLine object): legend lines to plot lines of upper plot
            dict (2DLine object): legend lines to plot lines of lower plot
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

        self.fill_below()
        return leg, lined, lined2

    def widget_init(self):
        """
        Initialize all widget objects.
        Returns:
            tuple (Widget Objects): span selector, rectangle selector, cursor,
                                    checkbox, slider, button
        """
        span = SpanSelector(self.ax, self.onselect, 'horizontal', useblit=True, minspan=5,
                            rectprops=dict(alpha=0.5, facecolor='red'))
        rect = RectangleSelector(self.ax, self.onselect_rect, drawtype='box', useblit=True,
                                 rectprops=dict(alpha=0.5, facecolor='red'))
        cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)

        cursor.active, span.active, rect.active = False, False, False
        plt.subplots_adjust(left=0.2, hspace=0.5)
        axc = plt.axes([0.05, .45, .1, .10]) # will have to adjust depending on span_name/rect_name
        axs = plt.axes([0.25, .48, .2, .03])
        axb = plt.axes([0.05, .05, .1, .10])
        # axp = plt.axes([0.025,.85,.13,.13])

        check = CheckButtons(axc, (self.span_name, self.rect_name), (False, False))
        check.on_clicked(self.check_function)
        button = Button(axb, 'Quit')
        button.on_clicked(self._quit)

        nicotine = 2.41
        slider = Slider(axs, 'Multiplier', 0.1, 5, valinit=nicotine)
        slider.on_changed(self.slider_update)
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())

        return span, rect, cursor, check, slider, button

    def connect(self):
        """
        Connects canvas of figure to selection events.
        """
        plt.connect('pick_event', self.onpick)
        plt.connect('button_press_event', self.cursor_off)
        plt.connect('button_release_event', self.cursor_on)
        self.button.on_clicked(self._quit)

    def onpick(self, event):
        """
        On a PickEvent, toggle lines, respective legend lines in both
        the upper and lower plots.
        Args:
            event (PickEvent): the artist of selected item of the plot
        """
        # find plot line corresponding to legend proxy line
        legline = event.artist
        ch = int(str(legline)[-2])
        self.visible[ch] = not self.visible[ch]
        origline = self.lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)

        origline2 = self.lined2[legline] # for zoomed in region
        vis = not origline2.get_visible()
        origline2.set_visible(vis)
        # Change the alpha on the legend proxy line so we can see what lines
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
        available = [i for i in xrange(9) if self.live[i] * self.visible[i]]
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

        y2 = np.array([self.lethal] * final_len)

        if len(self.ax2.collections): # avoid overwriting fill_betweens
            self.ax2.collections.pop()
        if len(self.ax.collections):
            self.ax.collections.pop()

        self.ax2.fill_between(tfill, np.array(yfill), facecolor='#F73C3C',
                              where=yfill > y2)#,alpha=0.3)
        self.ax.fill_between(tfill, np.array(yfill), facecolor='#F73C3C',
                             where=yfill > y2)#,alpha=0.3)
        self.fig.canvas.draw()

    def onselect(self, xmin, xmax):
        """
        Adjusts view of lower plot by changing the x-axis limits and redrawing.
        Args:
            xmin (int): lower x-axis bound of selected region
            xmax (int): upper x-axis bound of selected region
        """
        x = self.all_times
        _, indmax = np.searchsorted(x, (xmin, xmax))
        indmax = min(len(x) - 1, indmax)
        thisy_min, thisy_max = float('inf'), -float('inf')

        for i in (i for i in xrange(9) if self.live[i]):
            data = max(self.lines2[i].get_ydata())
            thisy_min = min(thisy_min, data)
            thisy_max = max(thisy_max, data)

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

    def onselect_rect(self, eclick, erelease):
        """
        Adjusts view of lower plot by changing the x-axis and y-axis limits and redrawing.
        Args:
            eclick (MousePressEvent): object representing point on the plot
                                      where the mouse is first pressed
            erelease (MouseReleaseEvent): object representing point on the plot
                                          where the mouse is released
        """
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

    def check_function(self, label):
        """
        Toggles cursor animation with chosen selector.
        Args:
            label (Button): the desired selector
        """
        if label == self.span_name:
            self.span.active = not self.span.active
        elif label == self.rect_name:
            self.rect.active = not self.rect.active
            self.cursor.active = self.rect.active
        self.fig.canvas.draw()

    def cursor_off(self, event):
        """
        Toggle visibility of cursor.
        Args:
            event (MousePressEvent): object representing click of mouse
        """
        if self.rect.active:
            self.cursor.active = False

    def cursor_on(self, event):
        """
        Toggle visibility of cursor.
        Args:
            event (MousePressEvent): object representing click of mouse
        """
        if self.rect.active:
            self.cursor.active = True

    def slider_update(self, val):
        """
        Redraw both upper and lower plots with adjsuted multiplier.
        Args:
            val (float): the selected value of the slider
        """
        multiplier = self.slider.val
        ylim_max = 0
        xlim = self.ax2.get_xlim()
        ylim2 = 0
        x = self.all_times
        _, indmax = np.searchsorted(x, xlim)
        indmax = min(len(x) - 1, indmax)
        for i in (i for i in xrange(9) if self.live[i]):
            mult = [multiplier/self.init_mult*num for num in self.y_data_complete[i]]
            ylim_max = max(ylim_max, max(mult))
            ylim2 = max(ylim2, max(mult[:indmax]))
            self.lines[i].set_ydata(mult)
            self.lines2[i].set_ydata(mult)
        if self.agent != None:
            self.ax.axvline(x=self.agent, color='b', linestyle='--')
            self.ax2.axvline(x=self.agent, color='b', linestyle='--')

        self.ax.set_ylim(-0.5, ylim_max + 1)
        self.ax2.set_ylim(-0.5, ylim2 + 1)
        self.fill_below()

    def _quit(self, event):
        """
        Quits out of program.
        Args:
            event (MousePressEvent): object representing click of mouse
        """
        raise SystemExit
