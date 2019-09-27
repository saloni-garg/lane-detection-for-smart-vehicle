from laneDetectionMethods import *

#Test on a single image
test_images = [read_image('test_images/' + i) for i in os.listdir('test_images/')]
print(test_images[0].shape)
draw_lane_lines(test_images[0])
plt.show()