RIF Sensor Python Application and Data Management 


~~~~~~~~~ File Overview ~~~~~~~~~
- Applications (Mac)
	- colorbar_only.app*
		Simple application to demonstrate standalone support for Matplotlib
	- 'Static RIF Sensor.app'*
		Uses previously collected data to simulate live plotting, includes final graph
		Functions:
			Live Graph:
				- toggling visibility of lines
				- Real Time Clock (RTC)
				- Fill below designated "lethal level"
			Final Graph:
				- toggling visibility of lines
				- Fill below designated "lethal level"
				- [span/rectangle] Selector to observe specific regions of the graph
				- Slider to adjust constant factor


- Current Build
	- analog_plotter.py
		Plots serial data gathered via hardwired Arduino
	- /build, /dist
		Used for creating the Mac application (the application can be found in /dist)
	- datatest.csv (and variants)
		Previously gathered data to simulate live plotting to test new functionalities
	- macholib_patch.py
		Necessary for packaging Matplotlib into Mac application
	- new_data.py
		Sample data generator (uses datatest.csv and multiplies by constant factor)
	- SensorFinal.py
		Static plotter when data stream has ended
		Functions:
			- toggling visibility of lines
			- Fill below designated "lethal level"
			- [span/rectangle] Selector to observe specific regions of the graph
			- Slider to adjust constant factor
	- SensorLive.py
		Dynamic plotter for live incoming data stream
		Functions:
			- toggling visibility of lines
			- Fill below designated "lethal level"
	- settings.py
		Main settings for both sensor plotters
	- setup.py
		Necessary for packaging Matplotlib into Mac application
	- subplot_test.py
		Newest implementation of plotter, creates subplots to allow for thumbnail images of other sensors
	- test.py
		Previous test of plotter


- Data Samples and Storage
	- datatest.csv
		Previously gathered data to simulate live plotting to test new functionalities (same as mentioned above)


- Examples and Test Files
	- fill_line_test.py
		Tests backfilling capabilities to account for discrete time measurements
	- sample_dynamic_plotter.py
		Example dynamic plotter using the Pyqtgraph library (Goal for next iteration)


- Previous Builds
	Includes various .py files from past iterations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~ EXAMPLE OF FUNCTIONALITY ~~~~~~~~~
- SensorLive and SensorFinal
	For complete descriptions of each method, read the doc strings in the file

	example:

		from SensorLive import *        #look into settings.py for clarification on what is
		from SensorFinal import * 		# being used exactly
								 
		import csv				 # for reading datatest.csv

		sensor = LiveSensor("Sensor Name")

		# iterate through data source, in this case, we'll use datatest.csv
		reader = csv.reader(open('datatest.csv', 'rU'))

		for row in reader:
			if not sensor.stop:       			# The sensor is still functional
				try:
					sensor.update_data(row)

				except KeyboardInterrupt:		# ctrl-c
					print '\n'
					break

		data = sensor.export_data()
		final_sensor = SensorFinal('Final Sensor Name', data)
		final_sensor.connect()                  # connect sensor to interactive events

		sensor.live_off()                       # same as plt.show()

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~ SET UP FOR DEVELOPMENT ~~~~~~~~~~
	These are critical steps for continuiung the coding process, and understanding what has been written thus far.

	KNOWLEDGE:
		1. Understand basic Python coding, including class structure, data manipulation, and basic algorithms.
		2. Understand the Python matplotlib library and its tendencies to prefer graphics over performance (and how to craftily improve the latter without affecting the former).
		3. Take time to use the example applications and run the example code. Understand what each method does and see how it comes together.

	SETUP:
		1. Install Python 2.x, if it isn't already from https://www.python.org/downloads/ 
		2. Install a text editor intended for coding (Sublime Text is great)
			2a. If using Sublime, install and configure sublime-linter and pylint. There are many sources online to help you with this (Pylint is used for code structure and clarity).

	GOALS:
		1. Improve the conditional blitting of the live sensor plot
		2. Handle and store data efficiently and effectively
		3. Improve overall plot performance (including speed and data management)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* for learning, not functional purpose