from sortedcontainers import SortedDict, SortedList
from collections import defaultdict, deque, OrderedDict



class LiveData(object):
    def __init__(self, live, visible):
        self.data = {i:[deque([]), [], []] for i in xrange(9)} # Channel: [[viewable], [full], [complete]]
        self.all_data_live = SortedDict() # Timestamp: SortedDict{channel:resistance}
        self.all_data_dead = SortedDict() # Timestamp: SortedDict{channel:resistance}
        self.max_indices = {} # Channel: Timestamps with max value of Channel
        self.live = live
        self.visible = visible
        self.changed = [] # list of potentially changed lines visibility

    def edit_viewable(self, channel):
        """"
        scroll current data to conserve memory
        """
        pass

    def update_channel(self, channel, point):
        """
        update data per channel
        """
        x_data, y_data = point
        # new_time = SortedDict({channel:point})

        if not self.live[channel]:
            # self.all_data_dead[channel].append(y_data)
            entry = self.all_data_dead.setdefault(channel, SortedList(key=lambda x: -x))
        else:
            # self.all_data_live[channel].append(y_data)
            entry = self.all_data_dead.setdefault(channel, SortedList(key=lambda x: -x))

        entry[channel] += [y_data]

        if len(self.data[channel][0]) > 50: # maintain buffer size
            self.data[channel][0].popleft()

        self.data[channel][0].append(y_data) # update viewable data
        self.data[channel][1].append(y_data) # update full data

        if len(self.changed) > 0:
            for channel in self.changed:
                self.update_max_indices(channel)
        pass

    def update_max_indices(self, channel):
        """
        everytime visibility is affected, invoke this function
        """
        if len(self.max_indices[channel]) > 0:
            # update all_data_live
            # update all_data_dead
            pass

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
