# Lane Detection For Smart Vehicle Using OpenCV on Raspberry Pi 3b+

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