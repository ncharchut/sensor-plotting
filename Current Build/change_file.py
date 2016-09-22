import csv

def modify_file(number):

    reader = csv.reader(open('recovery%d.csv' % (number), 'rU'))
    writer = open('middle.txt', 'w')

    for line in reader:
        string = line
        new_string = ""
        for char in string:
            if char == ',':
                continue
            new_string += char
        writer.write(new_string)
        writer.write('\n')

    new_reader = open('middle.txt', 'rU')
    writer = csv.writer(open('adjusted%d.csv' % (number), 'w'))

    for row in new_reader:
        string = row.split(' ')
        if string == ['\n']:
            continue
        print string
        channel = string[0]
        timestamp = string[4]
        resistance = string[6][:-1]
        data = [channel, timestamp, resistance]

        writer.writerow(data)

