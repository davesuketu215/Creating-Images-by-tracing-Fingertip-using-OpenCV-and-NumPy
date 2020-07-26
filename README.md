# Image_By_Fingertip_Trace
This is my first Computer Vision project using OpenCV and NumPy.
Whatever the user will draw in the free air, within the "Region of Trace" on the screen, it will be captured through WebCam and traced onto the whiteboard.
Color Palette is available to select various colors for drawing on the whiteboard.
The final trace image from the whiteboard will be downloaded as a ".jpg" file, with a timestamp.

# Concepts used:
1. Image Segmentation
2. Background Subtraction
3. Convex Hull
4. Some School-Level Mathematics

# Files:
main.py: The main project file with the tracing logic
color_palette: Returning an instance of color box

# Note:
1. Just run the main.py file.
2. Ensure a uniform background. Otherwise the background subtraction algorithm fails.
3. Results might vary with varying illumination levels in the room. Hence, ensure you are in a properly illuminated room.

# Demo Video Link:
YouTube: https://www.youtube.com/watch?v=WRE9eQk_tqo