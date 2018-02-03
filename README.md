# Stereo-Vision

This program has been developed as part of a project at the University of Karlsruhe in Germany. The final purpose of the algorithm is to measure the distance to an object by using Stereo Cameras.

A video of the project can be found here: https://www.youtube.com/watch?v=xjx4mbZXaNc

It is still an experimental version and further improvements will have to be done.

## Contributors
- Stephane VUJASINOVIC
- Frederic UHRWEILLER

## Requirements

You will need:
- Two webcams
- Python installled on your computer
- A Chessboard for the calibration

Required python packages:
- numpy
- openCV
- openpyxl
- scikitlearn

## "Take_images_for_calibration.py"

You will need to take pictures with both cameras yith this script in order to calibrate the cameras (remove the distorsion and stereo calibration)
It is good to take around 30 to 50 pictures with both cameras.

## ”Main_Stereo_Vision_Prog.py”

This script is used to calibrate the cameras with the pictures previously taken by the script "Take_images_for_calibration.py". 
A disparity map is then generated and thank to a third order regression, it is possible to measure the distance to a defined object by clicking on it. 
