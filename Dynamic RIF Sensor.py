#!/usr/bin/env python
import matplotlib
matplotlib.use('TkAgg')
import scipy, numpy as np 
from drawnow import drawnow # allows for fast live updating
import matplotlib.pyplot as plt
from collections import deque
import csv
from math import log
import time
from matplotlib.widgets import  Button, CheckButtons, Cursor, RectangleSelector, Slider, SpanSelector
import itertools
from matplotlib.font_manager import FontProperties
from scipy.misc import imread
import matplotlib.cbook as cbook
from datetime import datetime
import Tkinter
import FileDialog
import subprocess

basetime = True
channel_count, r0_step = 0, 0
lines, lined = [], {}
colors = ['b','g','r','c','m','y','k','purple','darkgreen']

channels = [[0] for i in xrange(9)]
channels_dup = [[0] for i in xrange(9)]
time_x = [[0] for i in xrange(9)]
time_x_dup = [[0] for i in xrange(9)]
channels_full = [[0 for i in xrange(i)] for i in xrange(9)]

font = FontProperties()
font.set_family('serif')
matplotlib.rc('font', family='serif')

strchannels = ['ch'+str(i) for i in xrange(9)]
channelindices = {strchannels[i]:i for i in xrange(9)}
stop = False
agent, agent_name = None, 'Agent Introduced'
r0 = {i:0.3 for i in xrange(9)}
ylim_max = 0
visible = {k:True for k in xrange(9)}
live = {i:True for i in xrange(9)}
live[1] = live[2] = False
threshold = -.01
init_mult = 2.41
lethal = 5
maxy, maxline, ind= 0, 1, 0

adj = lambda r,ch: init_mult*log(float(r)/r0[ch]) if r > 0 else r
chval = lambda ch: channels[ch]

def onpick(event):
    # on the pick event, find the orig line corresponding to the
    # legend proxy line, and toggle the visibility
    legline = event.artist
    ch = int(str(legline)[-2])
    visible[ch] = not visible[ch]

def onpick_f(event):
    # on the pick event, find the orig line corresponding to the
    # legend proxy line, and toggle the visibility
    legline = event.artist
    ch = int(str(legline)[-2])
    visible[ch] = not visible[ch]
    origline = lined[legline]
    vis = not origline.get_visible()
    origline.set_visible(vis)

    origline2 = lined2[legline] # for zoomed in region
    vis = not origline2.get_visible()
    origline2.set_visible(vis)
    # Change the alpha on the line in the legend so we can see what lines
    # have been toggled
    if vis:
        legline.set_alpha(1.0)
    else:
        legline.set_alpha(0.2)

    fill_below()

def fill_below():
    maxlines =  [lines2[j].get_ydata() for j in (i for i in xrange(9) if live[i]*visible[i])]
    maxtimes = [lines2[j].get_xdata() for j in (i for i in xrange(9) if live[i]*visible[i])]
        
    if len(maxlines) == 0:
        if len(ax2.collections):
            ax2.collections.pop()
        if len(ax.collections):
            ax.collections.pop()
        return fig.canvas.draw()

    final_len = len(maxlines[-1]) 

    for linea in maxlines:
        if len(linea) != final_len:
            linea.pop()

    yfill = [0 for i in xrange(final_len)]
    tfill = [0 for i in xrange(final_len)]
    llines, ltimes = list(zip(*maxlines)), list(zip(*maxtimes))

    for i in xrange(final_len):  
        yfill[i] = max(llines[i])
        tfill[i] = ltimes[i][llines[i].index(yfill[i])]

    y2 = np.array([lethal]*final_len)

    if len(ax2.collections):
        ax2.collections.pop()
    if len(ax.collections):
        ax.collections.pop()
    ax2.fill_between(tfill,np.array(yfill), facecolor='#F73C3C',where=yfill > y2)#,alpha=0.3)#, interpolate=True)
    ax.fill_between(tfill,np.array(yfill), facecolor='#F73C3C',where=yfill > y2)#,alpha=0.3)#, interpolate=True)
    fig.canvas.draw()

def stop_func(event):
    if event.key == ' ':     # generates vertical line at spacebar press
        global agent
        agent = all_times[-1] if agent == None else agent
    if event.key == 'ctrl+z': # stops live graph generation
        global stop
        stop = True
        p.terminate()

def plotfunc():
    plt.title('Sensor Resistance v. Time',fontsize=28)
    plt.ylabel('$ng/mm^2$',fontsize=20)
    plt.xlabel('Time [ms]',fontsize=20)
    plt.tick_params(axis='both', labelsize=15)
    lines = []
    
    global maxy
    global maxline
    global ind
    maxy, maxline, ind= 0, 1, 0
    ax = plt.gca()


    # adj = lambda r,ch: init_mult*log(float(r)/r0[ch]) if r > 0 else r
    # chval = lambda ch: channels[ch]

    for i in (i for i in xrange(9) if live[i] and adj(chval(i)[-1],i) > threshold):
        ch = strchannels[i]
        r0_adjusted = [adj(r,i) if r > 0 else 0 for r in chval(i)]
        if visible[i]:  # identify line with greatest current value
            temp = maxy
            maxy = max(maxy, max(r0_adjusted))
            maxline = r0_adjusted if temp != maxy else maxline
            ind = i if temp != maxy else ind
        line, = plt.plot(time_x[i],r0_adjusted,color=colors[i],label=ch,linewidth=2.0) # UPDATE EACH CHANNEL
        lines.append(line)
        if agent != None:
            plt.axvline(x=agent,color='b',linestyle='--')

    if maxline != 1:    # fill from maxline and the x axis
        y2 = np.array([lethal for i in maxline])
        for fill in ax.collections:
            ax.collections.pop()
        plt.fill_between(time_x[ind],maxline, facecolor='green',alpha=0.3,where=maxline > y2)
        
    leg = plt.legend(fancybox=True,shadow=True,loc='upper left',framealpha=0.5,prop=font)
    leg.get_frame().set_alpha(0.4)
    leg.get_frame().set_facecolor('LightGreen')

    for legline, origline in zip(leg.get_lines(), lines):  # toggle line visibility with legend
        legline.set_picker(7)  # 7 pts tolerance
        lined[legline] = origline
        ch = int(str(legline)[-2])
        vis = visible[ch]
        origline.set_visible(vis)
        alpha = 1.0 if vis else 0.2
        legline.set_alpha(alpha)

    axp = plt.axes([0.75,.89,.13,.13])
    axp.axis('off')
    timestamp = 'Local time: ' + str(datetime.time(datetime.now()))[:-5]
    text = axp.text(0.5, 0.5, timestamp,weight='bold',fontsize=15,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
    
    # ax = plt.gca()
    # for line in ax.lines:
    #     ax.lines.pop()
    # # text.remove()
    # plt.show()

###################### LIVE GRAPH ###########################  

# back fill resistance values at every timestep
# necessary to avoid glitches in fill_between ()
def fill_line(vals, interval,channel):
    startx, endx = interval[0],interval[-1]
    filled_line = [0 for i in xrange(9)]

    if not (endx-startx):
        return filled_line

    startval, endval = vals[channel][startx], vals[channel][endx]
    slope = float(endval-startval)/(endx-startx)
    
    filled_line[0] = startval
    for i in xrange(1,9):
        delta_x = interval[i]-interval[i-1]
        filled_line[i] = filled_line[i-1]+slope*delta_x

    return filled_line


vals = {i:{0:0} for i in xrange(9)}
f = open('data.csv', 'wb')
writer = csv.writer(f)
all_times = []
count = [0 for i in xrange(9)]
init_count = 0
exe = ['telnet','192.168.0.104']#'sense01.local']
p = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(3) # may not be necessary


while True:
    if not stop:
        plt.connect('pick_event', onpick)
        plt.connect('key_press_event', stop_func)
        retcode = p.poll() #returns None while subprocess is running
        data = p.stdout.readline() # takes a line of incoming data
        if retcode is not None:
            print """Connection unsuccessful. Please restart the sensor
                    and wait one minute before attempting connection."""
            raise SystemExit
    
        if init_count < 3:  # avoid the first three lines of output, they are irrelevant
            init_count += 1 
            continue 
    
        dataline = data
        dataArray = dataline.split(' , ') # might have to replace depending on delimiter
        channel,curtime,resistance = dataArray[0],dataArray[1],dataArray[2].split(' ')[0]
        index, curtime, resistance = int(channel), int(curtime), float(resistance)

        if basetime == True:
            basetime = curtime
        
        oldtime = curtime
        curtime -= basetime
        writer.writerow([channel, resistance, curtime, oldtime])

        channels[index].append(resistance)
        time_x[index].append(curtime)
        channels_dup[index].append(resistance)
        time_x_dup[index].append(curtime)    
        channel_count += 1

        all_times.append(curtime)

        vals[index][curtime] = resistance
        count[index] += 1

        if count[index] >= 2:
            interval = all_times[-10:]
            filled_line = fill_line(vals,interval,index)
            channels_full[index].extend(filled_line)
    
        if resistance != 0 and r0[index] != 0.3:
            ylim_max = max(ylim_max, init_mult*log(resistance/r0[index]))

        if channel_count == 9:    # Needed to simulate live datastream
            r0_step += 1
            if r0_step >= 3 and r0[index] == 0.3: # determines r0 for all channels, adjust r_step for actual live sensor
                for i in xrange(9):
                    if chval(i)[-1] != 0:
                        r0[i] = chval(i)[-1]
            channel_count = 0
        drawnow(plotfunc)
        plt.pause(.001)

        ax = plt.gca()
        for line in ax.lines:
            ax.lines.pop()
        for txt in ax.texts:
            txt.remove()


    else:
        break
#############################################################

####################### SPAN SELECTOR #######################
def onselect(xmin, xmax):
    x = all_times # arbitrary selection from time_x_dup
    indmin, indmax = np.searchsorted(x, (xmin, xmax))
    indmax = min(len(x) - 1, indmax) 
    # thisx_min, thisx_max = float('inf'),-float('inf')
    thisy_min, thisy_max = float('inf'),-float('inf')

    for i in (i for i in xrange(9) if live[i]):
        data = lines2[i].get_ydata()
        thisy_min = min(thisy_min,data[indmin])
        thisy_max = max(thisy_max, data[indmax])

        # If data sets are huge, it may be necessary to slice
        # This could cause issues in general widget cohesiveness

        # xdata = list(itertools.islice(time_x_dup[i],max(0,indmin-5),min(indmax+5,len(x)-1)))
        # ydata = list(itertools.islice(channels_dup[i],max(0,indmin-5),min(indmax+5,len(x)-1)))
        # lines2[i].set_data(xdata,ydata)

    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(thisy_min-2, thisy_max+2) # window cushion
    fig.canvas.draw()
#############################################################

##################### RECTANGLE SELECTOR #################### 
def onselect_rect(eclick, erelease):
    x_init, y_init = eclick.xdata, eclick.ydata
    x_final, y_final = erelease.xdata, erelease.ydata  

    # If data sets are huge, it may be necessary to slice
    # This could cause issues in general widget cohesiveness

    # xmin, xmax = min(x_init,x_final), max(x_init, x_final)
    # ymin, ymax = min(y_init,y_final), max(y_init, y_final)

    # x = time_x_dup[0] # arbitrary selection from time_x_dup
    # indmin, indmax = np.searchsorted(x, (xmin, xmax))
    # indmax = min(len(x) - 1, indmax)

    # for i in xrange(9):
    #     if not lineindices[i]: # blacked out lines
    #         continue
    #     xdata = list(itertools.islice(time_x_dup[i],max(0,indmin-5),min(indmax+5,len(x)-1)))
    #     ydata = list(itertools.islice(channels_dup[i],max(0,indmin-5),min(indmax+5,len(x)-1)))
    #     lines2[i].set_data(xdata,ydata)

    ax2.set_xlim(x_init, x_final)
    ax2.set_ylim(y_init, y_final) # window cushion
    fig.canvas.draw()
#############################################################

############## CHECKBUTTON FOR GRAPH SELECTOR ###############
def func(label):
    if label == span_name:
        span.active = not span.active
    elif label == rect_name:
        rect.active = not rect.active
        cursor.active = rect.active
    plt.draw()
#############################################################

##################### CURSOR VISIBILITY #####################
def cursor_off(event):
    if rect.active:
        cursor.active = False

def cursor_on(event):
    if rect.active:
        cursor.active = True
#############################################################

##################### SLIDER FUNCTION #######################
def update(val):
    multiplier = slider.val
    ylim_max = 0
    xlim = ax2.get_xlim()
    ylim2 = 0
    x = all_times # arbitrary selection from time_x_dup
    indmin, indmax = np.searchsorted(x, xlim)
    indmax = min(len(x) - 1, indmax) 
    for i in (i for i in xrange(9) if live[i]):
        mult = [multiplier/init_mult*num for num in channels_dup[i]]
        ylim_max = max(ylim_max, max(mult))
        ylim2 = max(ylim2,max(mult[:indmax]))
        lines[i].set_ydata(mult)
        lines2[i].set_ydata(mult)
    if agent != None:
        ax.axvline(x=agent,color='b',linestyle='--')
        ax2.axvline(x=agent,color='b',linestyle='--')

    ax.set_ylim(-0.5,ylim_max+1)
    ax2.set_ylim(-0.5,ylim2+1)
    fill_below()
#############################################################

#################### STATIC FINAL GRAPH #####################
f.close()
fig = plt.figure(dpi=100,facecolor='#9EC4A8') # figsize=(24, 18))
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

lines, lines2 = {}, {}

ax.set_title('Sensor Resistance v. Time',fontsize=28,fontproperties=font)
ax.set_ylabel('$ng/mm^2$',fontsize=20)
ax.set_xlabel('Time [ms]',fontsize=20)
ax.set_ylim(-0.5,ylim_max+1)
ax.tick_params(axis='both', labelsize=15)

ax2.set_ylabel('$ng/mm^2$',fontsize=20)
ax2.set_xlabel('Time [ms]',fontsize=20)
ax2.set_ylim(-0.5,ylim_max+1)
ax2.tick_params(axis='both', labelsize=15)

for i in (i for i in xrange(9) if live[i]):
    ch = strchannels[i]
    channels_dup[i] = [init_mult*log(float(r)/r0[i]) if r > 0 else 0 for r in channels_full[i]]
    # channels_dup[i] = channels_dup[i][:len(all_times)]
    if chval(i)[-1] < 0: # if last value is negative, disregard the associated line
        live[i] = False
        continue

    r =  channels[i][-1] # last adjustments to final filled lines
    channels_dup[i].append(init_mult*log(float(r)/r0[i]))

    while len(channels_dup[i]) != len(all_times):
        channels_dup[i].append(channels_dup[i][-1])

    line, = ax.plot(all_times,channels_dup[i],color=colors[i],label=ch,linewidth=2) # UPDATE EACH CHANNEL
    line2, = ax2.plot(all_times,channels_dup[i],color=colors[i],label=ch,linewidth=2)
    lines[i] = line
    lines2[i] =line2

if agent != None: # vertical lines
        ax.axvline(x=agent,color='b',linestyle='--')
        ax2.axvline(x=agent,color='b',linestyle='--')

leg = ax.legend(loc='upper left', fancybox=True, shadow=True)
leg.get_frame().set_alpha(0.4)
leg.get_frame().set_facecolor('#83B389') # greenish, kinda
lined, lined2 = {}, {}

for legline, origline, origline2 in zip(leg.get_lines(), lines.values(), lines2.values()):
    legline.set_picker(7)  # 7 pts tolerance
    lined[legline] = origline
    lined2[legline] = origline2
    ch = int(str(legline)[-2])
    vis = visible[ch]
    origline.set_visible(vis)
    origline2.set_visible(vis)
    alpha = 1.0 if vis else 0.2
    legline.set_alpha(alpha)

fill_below()

##########################################################

span = SpanSelector(ax, onselect, 'horizontal', useblit=True, minspan=5,
                    rectprops=dict(alpha=0.5, facecolor='red'))
rect = RectangleSelector(ax, onselect_rect, drawtype='box',useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'))
cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

span_name, rect_name = 'Span', 'Rect'
cursor.active, span.active, rect.active = False, False, False
plt.subplots_adjust(left=0.2,hspace=0.5)
axc = plt.axes([0.05,.45,.1,.10]) # will have to adjust depending on span_name/rect_name
axs = plt.axes([0.25,.48,.2,.03])
axp = plt.axes([0.025,.85,.13,.13])

check = CheckButtons(axc, (span_name, rect_name) , (False, False))
check.on_clicked(func)

nicotine = 2.41
slider = Slider(axs, 'Multiplier', 0.1, 5, valinit=nicotine)
slider.on_changed(update)

plt.connect('pick_event', onpick_f)
plt.connect('button_press_event', cursor_off)
plt.connect('button_release_event', cursor_on)
#############################################################
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
    
datafile = 'logo.png'
img = imread(datafile)
axp.imshow(img)
axp.axis('off')

plt.show()
