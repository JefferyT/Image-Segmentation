# deep-learning-final-project

Project:
We want to create a project that will perform semantic-segmentation on images in a way that the model can identify between background and foreground of an image to enable image manipulation like filters or blurring the background. After we can apply this to images, then if time allows, we want to apply this to videos as well.

Approach:
Our approach will be to preprocess the images, create a model for semantic segmentation using a convolution network and a deconvolution network. Then, after choosing the segment to focus, apply a filter effect over the image.


Datasets:
We will try out a few of the datasets found at this link:
https://paperswithcode.com/datasets?task=semantic-segmentation
We will likely start by using the VOC dataset and see how our model performs on that before attempting some other datasets.

Evaluate Result:
Accuracy of model against test set.
Since the final result will be an image/video human evaluation can be done as well.
