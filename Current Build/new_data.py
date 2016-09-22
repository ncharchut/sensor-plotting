from __future__ import print_function
import csv

reader = csv.reader(open('datatest.csv', 'rU'))
writer=csv.writer(open('datatest3.csv', 'w'))

for row in reader:
    channel, curtime = int(row[0]), int(row[1])
    resistance = float(row[2].split(' ')[0])*1.8
    new_row = [channel, curtime, resistance]
    # print(new_row, file=open('datatest1.csv', 'w'))
    writer.writerow(new_row)

print('done')
