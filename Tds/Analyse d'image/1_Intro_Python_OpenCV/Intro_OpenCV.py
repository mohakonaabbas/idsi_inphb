# import the necessary packages
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2

# debug info OpenCV version
print("OpenCV version: " + cv2.__version__)

# 1 - Basic I/O scripts ###########################################

# 1-1 Reading/writing an image file -------------------------------
img1 = np.zeros((3,3), dtype=np.uint8)
print('Shape of a gray level image')
print(img1.shape)
img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
print('Shape of a color image')
print(img1.shape)

#    Read and write in an another format
img2 = cv2.imread('data_images/other/lena.jpg')
cv2.imwrite('log/Lena.png', img2)

#    Read like a gray level image
grayImg1 = cv2.imread('data_images/other/lena.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('log/lenaGray.png', grayImg1)

# 1-2 Converting between an image and raw bytes ------------------

# Make an array of 120,000 random bytes.
randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = np.array(randomByteArray)

# Convert the array to make a 400x300 grayscale image.
grayImg2 = flatNumpyArray.reshape(300, 400)
cv2.imwrite('./log/RandomGray.png', grayImg2)

# Convert the array to make a 400x100 color image.
bgrImg1 = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite('./log/RandomColor.png', bgrImg1)

# 1-3 Accessing image data with numpy.array ----------------------
img3 = cv2.imread('data_images/other/lena.jpg')
img3[0,0] = [255, 255, 255]
print('Value of B for the pixel (150, 120)')
print(img2.item(150, 120, 0)) # prints the current value of B for that pixel
# img3.itemset( (150, 120, 0), 255)
print('New value of B for the pixel (150, 120)')
print(img3.item(150, 120, 0))

my_roi = img3[0:100, 0:100]
img3[300:400, 300:400] = my_roi
cv2.imwrite('log/lenaROI.png', img3)

# 1-4 Reading/writing a video file -------------------------------
videoCapture1 = cv2.VideoCapture('data_images/other/Echo.avi')
fps = videoCapture1.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture1.get(cv2.CAP_PROP_FRAME_WIDTH)),int(videoCapture1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter1 = cv2.VideoWriter('./log/MyOutputVid1.avi', cv2.VideoWriter_fourcc('I','4','2','0'),fps, size)
success, frame = videoCapture1.read()
while success: # Loop until there are no more frames.
    videoWriter1.write(frame)
    success, frame = videoCapture1.read()

# 1-5 Capturing camera frames ------------------------------------
cameraCapture1 = cv2.VideoCapture(0)
fps = 30 # an assumption
size = (int(cameraCapture1.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cameraCapture1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter2 = cv2.VideoWriter('./log/MyOutputVid2.avi', cv2.VideoWriter_fourcc('I','4','2','0'),fps, size)
success, frame = cameraCapture1.read()
numFramesRemaining = 10 * fps - 1
while success and numFramesRemaining > 0:
    videoWriter2.write(frame)
    success, frame = cameraCapture1.read()
    numFramesRemaining -= 1
cameraCapture1.release()

# 1-6 Displaying images in a window ------------------------------
cv2.imshow('my image', img3)
cv2.waitKey()
cv2.destroyAllWindows()

# 1-7 Displaying camera frames in a window -----------------------
clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True
cameraCapture2 = cv2.VideoCapture(0)
cv2.namedWindow('MyWindow')
cv2.setMouseCallback('MyWindow', onMouse)
print('Showing camera feed. Click window or press any key to stop.')
success, frame = cameraCapture2.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow('MyWindow', frame)
    success, frame = cameraCapture2.read()
cv2.destroyWindow('MyWindow')
cameraCapture2.release()

# 2 - Processing Images with OpenCV ##############################

# 2-1 The Fourier Transform --------------------------------------
img4 = cv2.imread("data_images/texture/texture_1.png", 0)
img4_float32 = np.float32(img4)
dft = cv2.dft(img4_float32,flags = cv2.DFT_COMPLEX_OUTPUT)
fshift = np.fft.fftshift(dft)
imgFFT = 20*np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1]))
plt.subplot(121),plt.imshow(img4, cmap = 'gray')
plt.title("Image")
plt.subplot(122),plt.imshow(imgFFT, cmap = 'gray')
plt.title("Spectrum of image")
plt.show()

# 2-2 Edge detection ---------------------------------------------
img5 = cv2.imread("data_images/other/lena.jpg", 0)
cv2.imwrite("./log/canny.jpg", cv2.Canny(img5, 100, 150))
cv2.imshow("canny.jpg", cv2.imread("./log/canny.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()

# 2-3 Contour detection ------------------------------------------
img6 = np.zeros((200, 200), dtype=np.uint8)
img6[50:150, 50:150] = 255
img6[175:190, 175:190] = 255
ret, thresh = cv2.threshold(img6, 127, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
color = cv2.cvtColor(img6, cv2.COLOR_GRAY2BGR)
img6 = cv2.drawContours(color, contours, -1, (0,255,0), 2)
print(len(contours))
cv2.imshow("contours", color)
cv2.waitKey()
cv2.destroyAllWindows()
