# Project *Go there Vector*
___
Project to experiment with: Flask, html, javascript, video streaming,OpenCV. <br>
Using robot Vector (DDL/Anki's Robot: https://www.digitaldreamlabs.com/products/vector-robot)
<br>

## Description
The web page display a live video of Vector's play area. The user can clic the page to have Vector
to move to that location. <br>
If Vector is *available*, its position and orientation are determined to calculate path to the desire point. 
Then commends are sent to Vector to move to this location. While moving, a second video streaming is opened to 
display Vector front camera. <br>

## Helpful methods/code
Miguel Grinberg Flask video streaming codes : https://github.com/miguelgrinberg/flask-video-streaming 
<br>
Examples of OpenCV usage: https://github.com/automaticdai/rpi-object-detection

## Current limitation / bug
Tests have been conducted only with Flask's development server <br>
Application is unable to disconnect from Vector remote control, possibly due to the video streaming of
Vector's camera in a background thread.  