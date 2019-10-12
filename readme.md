# Lane Detection For Smart Vehicle Using OpenCV on Raspberry Pi 3b+

## Running This Project:

**Step 1:** Getting setup with Python

For this project, you will need Python 3.5 along with the numpy, matplotlib, and OpenCV libraries.

It is reccomended to install the Anaconda Python 3 distribution from Continuum Analytics.

Choose the appropriate Python 3 Anaconda install package for your operating system <A HREF="https://www.continuum.io/downloads" target="_blank">here</A>.   Download and install the package.

If you already have Anaconda for Python 2 installed, you can create a separate environment for Python 3 and all the appropriate dependencies with the following command:

`>  conda create --name=yourNewEnvironment python=3.5 anaconda`

`>  source activate yourNewEnvironment`

**Step 2:** Installing OpenCV

Once you have Anaconda installed, first double check you are in your Python 3 environment:

`>python`
`Python 3.5.2 |Anaconda 4.1.1 (x86_64)| (default, Jul  2 2016, 17:52:12)`
`[GCC 4.2.1 Compatible Apple LLVM 4.2 (clang-425.0.28)] on darwin`
`Type "help", "copyright", "credits" or "license" for more information.`
`>>>`
(Ctrl-d to exit Python)

run the following command at the terminal prompt to get OpenCV:

`>  conda install -c https://conda.anaconda.org/menpo opencv3`

then to test if OpenCV is installed correctly:

`> python`
`>>> import cv2`
`>>>`
(Ctrl-d to exit Python)

**Step 3:** Installing moviepy

The "moviepy" package processes videos in this project (different libraries might be used as well).

To install moviepy run:

`>pip install moviepy`

and check that the install worked:

`>python`
`>>>import moviepy`
`>>>`
(Ctrl-d to exit Python)

**Step 4:** Run

Run `Image_processing.py` in order to detect the lane on a single image.
Run `Video_processing.py` in order to detect the lane on a video.


## Step-by-step explanation of the algorithm used:

### 1. Video Processing: The video footage frames will be sent with the speed of 100 frames per second. This can be reduced if the device being used has lower processing power.
The device we're using here is Raspberry Pi, which has processor clocked at 1.4 GHz.

### 2. Applying Canny Edge Detector:
The primary basis on which we're separating Lanes from their boundaries is the bright colour difference, considering the roads are generally gray-black in colour and the lane lines are white/yellow. 
This huge colour difference helps us identify lanes as lines in the video frames. This algorithm is designed to detect sharp changes in luminosity (large gradients), such as a shift from white to black, and defines them as edges, given a set of thresholds.


### 3. Lane Area Segmentation:
A triangular mask is prepared to segment the lane area and discard the irrelevant areas in the frame to increase the effectiveness at later stages.

### 4. Hough transform:
This feature extraction technique is used to extract the lanes as straight lines from the video frames, using the voting procedure.
Read more about Hough Transform here: https://www.researchgate.net/publication/272195556_A_Survey_on_Hough_Transform_Theory_Techniques_and_Applications

Since our frame passed through the Canny Detector may be interpreted simply as a series of white points representing the edges in our image space, we can apply the same technique to identify which of these points are connected to the same line, and if they are connected, what its equation is so that we can plot this line on our frame.
Generally, the more curves intersecting in Hough space means that the line represented by that intersection corresponds to more points. For our impxlementation, we will define a minimum threshold number of intersections in Hough space to detect a line. Therefore, Hough transform basically keeps track of the Hough space intersections of every point in the frame. If the number of intersections exceeds a defined threshold, we identify a line with the corresponding θ and r parameters.

### 5. Visualization and Export:
The lane is visualized as two light green, linearly fitted polynomials which will be overlayed on our input frame.

## To-do:
- [x] Complete Documentation
- [x] Include results obtained when a video is input.
- [x] Add pointers to related Research articles.

## Acknowledgement:
This projects uses OpenCV, Canny Edge Detection Algorithm, Hough Transoformation, and TensorFlow.

## Contact
If you have any problems in reproducing the results, just raise an issue in this repo.