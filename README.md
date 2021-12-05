# Background-Subtraction
Background subtraction is a problem in computer vision where we wish to seperate the background from the image. It is also termed as Foreground Detection.

Background subtraction is a well-known technique for extracting the foreground objects in images or videos. The main aim of background subtraction is to separate moving object foreground from the background in a video, which makes the subsequent video processing tasks easier and more efficient. Usually, the foreground object masks are obtained by performing a subtraction between the current frame and a background model. The background model contains the characteristics of the background or static part of a scene.

## Running Averages
Here we have used the concept of running average which works as follows – we form a background model (which is a function of time) formed by taking a set of frames and averaging it out so that the common background remains and the foreground in removed. 
Foreground = Image(t) – BackgroundAverge(1 to t)
Finally, we have also used averaging filters after converting the image to gray scale.

## BackgroundKNN
It uses K nearest neighbors for background subtraction, where it divides the image into k clusters where each observation belongs to the cluster with nearest mean.

## BackgroundMOG
The Gaussian mixture model is a category of the probabilistic model which states that all generated data points are derived from a mixture of finite Gaussian distributions. MOG uses a method to model each background pixel by a mixture of K Gaussian distributions.

## How to run the code
Run the script a follows -
```
python main.py --i [input path] --o [output path]
```
Here the input path referes to the video sequence or folder containing images and the output path is where the masks must be stored.

## Evaluation Script
The python file eval.py evaluates the masks created based on IoU(Intersection Over Union).
Run the script a follows -
```
python eval.py --p [prediction path] --g [groundtruth path]
```
