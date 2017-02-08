# hough_transform
Detecting line and circle parameters via Hough Transformation

This Python script attempts to detect circle parameters in images without the use of iterative loops. To acheive this, the script makes extensive use of the NumPy library for manipulating image and Hough Accumulator matrices. Obviously, this was done to improve detection speed, however timing operations have not yet been performed since some optimizations still remain to be performed (which are listed at the top of the .py file). The code itself is heavily commented on, so there should be few issues reading through it. 

Note: An iterative 'for' loop is used at the end to draw the circles at the evaluated Hough Peaks. However, this is only for demonstration purposes and can be removed without affecting the utility of the program.
