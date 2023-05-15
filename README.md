# Creating Test Images and Assessing Segmentation Accuracy
This project focuses on testing and comparing the functionality of various image segmentation techniques and evaluating their accuracy. The objective is to work with images where the correct segmentation is known. The implementation involves generating test images with artificial objects on a constant gray-level background. Simple geometric objects such as squares, rectangles, diamonds, stars, and circles are created with varying gray levels different from the background. The area of each object is calculated and stored. Gaussian noise and random impulse noise are superimposed on the test images to create additional variations. The program includes methods to calculate segmentation accuracy indices, namely the "Relative Signed Area Error" and "Labeling Error," for quantitative comparison among different segmentation methods. The project also implements the basic thresholding and Chan-Vese segmentation methods and applies them to the test images created. The accuracy of the segmentation is assessed using the developed indices, allowing for a comparison of the performance of individual methods.
