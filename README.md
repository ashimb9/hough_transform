# hough_transform
Detecting line and circle parameters via Hough Transformation

There are two Python scripts, one of which (hough_lines.py) is for detecting straight lines in images and the other (hough_circle.py) is for detecting circle parameters in images. The circle detection script attempts to detect circle parameters without the use of iterative loops. To acheive this, the script makes extensive use of the NumPy library for manipulating the image and the Hough Accumulator matrix. Obviously, this was done to improve detection speed, however timing operations have not yet been performed since some optimizations still remain to be performed (which are listed at the top of the .py file). The code itself is heavily commented on, so there should be few issues reading through it. 

Note: An iterative 'for' loop is used at the end of hough_circles.py to draw the circles at the evaluated Hough Peaks. However, this is only for illustration purposes and can be removed without affecting the utility of the program.
