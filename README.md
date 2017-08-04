# webcam_face-eye_detect
Detect face and eyes 
Object Detection using Haar feature-based cascade classifiers is an effective object detection method.

Here we will work with face and eye detection. Initially, the algorithm needs a lot of positive images (images of faces and eyes) and negative images (images without faces and eyes) to train the classifier. Then we need to extract features from it. For this, haar features shown in below image are used. They are just like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels under white rectangle from sum of pixels under black rectangle.

![Rectangles](http://docs.opencv.org/trunk/haar_features.jpg)

But among all these features we calculated, most of them are irrelevant. So how do we select the best features out of 160000+ features? It is achieved by **Adaboost**.

OpenCV comes with a trainer as well as detector. If you want to train your own classifier for any object like car, planes etc. you can use OpenCV to create one. Its full details are given here: http://docs.opencv.org/trunk/dc/d88/tutorial_traincascade.html
