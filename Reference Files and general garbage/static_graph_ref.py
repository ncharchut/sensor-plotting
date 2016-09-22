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

# plt.show()