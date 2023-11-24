# Crossbreed Smarter Heating
What is Crossbreed Smarter Heating?

Crossbreed Smarter Heating is a optimization servicer that allows optimization of heat consumption up to 15%. This is done by using the building as a thermal storage and weather predictions. The service is available for both residential and commercial buildings.

It runs on Crossbreed Energy OS, _CEOS_, and is available as a service.

## Building Optimization
I have building that I want to optimize, what do I do?

You need to provide with the following information:

* What type of controlling equipment do you have? Siemens, Schneider, etc.
* What type of indoor climate sensors do you have? Wideco, Telia, Sensitive etc.
* How to access your equipment.

## Bootstrapping
What do I need to tell you in order to access my equipment?

You need to provide with the following information:

* The type of system and the identifiers for _CEOS_ to determine how to access it. Depending on the vendor it may be different.

## Siemens System
I have a siemens (climatixic) system, what do I need to provide?

You need to provide with the following information:

* The Tenant ID, for example: "T123".
* The Plant ID, for example "Plant 1".
* Which profile you want to use. Default is "generic".

## Indoor Sensors from Sensative
The indoor climate sensors are from Sensitive, what do I need to provide?

You need to provide with the following information:

* The sensor ID, for example: "S14".
* The Tenant ID, for example "myAccount".

## Indoor Sensors from Wideco
What do I need to provide in order to configure CEOS to get indoor data from Wideco Sensors?

You need to provide with the following information:
* The Account ID, for example: "myAccount".
* The Sensor ID, for example: "153933".
* If it is a indoor temperature sensor or a indoor humidity sensor.
* By default the sensor is assumed to be a temperature sensor.

## Installation and Positioning of Indoor Climate Sensors
How do I install the indoor climate sensors?

It is important that you install the indoor sensors in the right place. The sensors should be placed in the middle of the room, at a height of 1.5 meters. The sensors should not be placed in direct sunlight or near heat sources such as radiators or lamps. Do not place any sensors near windows or doors.

Place one sensor at the north wall and one at the south wall to be able to measure the temperature difference between the two sensors.