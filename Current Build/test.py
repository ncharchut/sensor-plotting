#!/usr/bin/env python
# pylint: disable=unused-import, unused-wildcard-import, no-self-argument, wildcard-import
# pylint: disable=too-many-instance-attributes, invalid-name, no-self-use, unused-argument

"""
Testing the plotters
-----------------
Simulates test, both from csv files and live data
"""

import csv
import time
import telnetlib
from SensorLive import *
from SensorFinal import *

reader = csv.reader(open('datatest.csv', 'rU'))
sensors = [SensorLive('sense0'+ str(num)) for num in xrange(1, 2)]

# for i in xrange(len(sensors)):
#     sensor = sensors[i]
old_time = 0
sensor = sensors[0]
sensor.connect()
initial_time = time.time()
for row in reader:
    if not sensor.stop:
        try:
            sensor.update_data(row)

        except KeyboardInterrupt:
            print '\n'
            break

# HOST = 'sense02.local'
# tn = telnetlib.Telnet(HOST)
# var = '?'

# while True:
#     if not sensor.stop:
#         try:
#             tn.write(var)
#             tn_read = str(tn.read_until(':\r\n'))
#             sensor.update_data(tn_read)

#         except KeyboardInterrupt:
#             print '\n'
#             break


final_time = time.time()
print 'This plotting took %d ms' % ((final_time - initial_time)*1000)
data = sensor.export_data()
final_sensor = SensorFinal('test', data)
final_sensor.connect()

sensor.live_off()
