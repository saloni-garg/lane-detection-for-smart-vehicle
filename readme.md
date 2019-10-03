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

Run `processImage.py` in order to detect the lane on a single image.
Run `processVideo.py` in order to detect the lane on a video.


## Step-by-step explanation of the algorithm used:

### 1. Video Processing: The video footage frames will be sent with the speed of 100 frames per second. This can be reduced if the device being used has lower processing power.
The device we're using here is Raspberry Pi, which has processor clocked at 1.4 GHz.

### 2. Applying Canny Edge Detector:
The primary basis on which we're separating Lanes from their boundaries is the bright colour difference, considering the roads are generally gray-black in colour and the lane lines are white/yellow. 
This huge colour difference helps us identify lanes as lines in the video frames. This algorithm is designed to detect sharp changes in luminosity (large gradients), such as a shift from white to black, and defines them as edges, given a set of thresholds.


### 3. Lane Area Segmentation:


### 4. Hough transform:


### 5. Visualization and Export:

## To-do:
- [ ] Complete Documentation
- [ ] Include results obtained when a video is input.
- [ ] Add pointers to related Research articles.

## Acknowledgement:
This projects uses OpenCV, Canny Edge Detection Algorithm, Hough Transoformation, and TensorFlow.

## Contact
If you have any problems in reproducing the results, just raise an issue in this repo.