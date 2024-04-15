# Gelsight Demo Basics

## Requirements (better creating a venv)

* opencv
* ros
* numpy
* matplotlib (for debug)
* mjpg installed and setup on the Raspberry Pi

## For code testing

* [lab_test.py](lab_test.py) and
[A_utility_test.py](A_utility_test.py) are used for testing and understanding using the existing database [Testdata](Testdata). 
* [lab_test.docx](lab_test.docx) and [A_utility_test.docx](A_utility_test.docx) are detailed code explains. 
* [Intermidiate output](Intermidiate%20output) contains all images outputs in the process of processing.

## Demo

* [lab_live_mjpg.py](lab_live_mjpg.py), [A_utility_live_mjpg.py](A_utility_live_mjpg.py) and [transformation_matrix_calculation.py](transformation_matrix_calculation.py) are used for performing real-time demo.
* When Pi starts streaming data through mjpg, run [transformation_matrix_calculation.py](transformation_matrix_calculation.py) and select four points clockwise (start from the top-left corner)

## Settings
Both code testing demo need [setting.py](setting.py) and [find_marker.so](find_marker.so), which can be compiled from the [Gelsight tracking project](https://github.com/Chengjindu/Gelsight_tracking_debug/tree/master)

**Set Parameters for Marker matching in [setting.py](setting.py)**

* RESCALE: scale down
* N, M: the row and column of the marker array
* x0, y0: the coordinate of upper-left marker (in original size)
* dx, dy: the horizontal and vertical interval between adjacent markers (in original size)
* fps_: the desired frame per second, the algorithm will find the optimal solution in 1/fps seconds

