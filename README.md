# Project *Go There Vector*
___
Project to experiment with: Flask, html, javascript, video streaming,OpenCV. <br>
Using robot Vector (DDL/Anki's Robot: https://www.digitaldreamlabs.com/products/vector-robot)
<br>

## Description
The web page display a live video of Vector's playground, fed from an overhead camera. The user can click within 
this area to have Vector to move to that destination. <br>
If Vector is *available*, its position and orientation are determined to calculate path to the desire point. 
Then commends are sent to Vector to move to this location. While moving, a second video window is opened to 
display Vector's front camera feed. <br>

Here is a demonstration video: <br>
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/KOITdRnjWs4/0.jpg)](https://www.youtube.com/watch?v=KOITdRnjWs4)


## Helpful methods/code
Miguel Grinberg Flask video streaming codes : https://github.com/miguelgrinberg/flask-video-streaming 
<br>
Examples of OpenCV usage: https://github.com/automaticdai/rpi-object-detection

## Current limitation / bug
Tests have been conducted only with Flask's development server <br>
Application is unable to disconnect from Vector remote control, possibly due to the video streaming of
Vector's front camera running in a background thread.  