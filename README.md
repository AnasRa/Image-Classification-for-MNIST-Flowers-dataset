# Project: Predicting Boston Housing Prices

In this project, I have trained image classifier to recognize 102 categories of flowers. Trained model is validated using inferencing on
validation image dataset. Model is finally tested on test dataset. Trained model is saved to checkpoint and later reloaded to predict on 
sample test images. Some of the image preprocessing is done to convert PIL image to tensor before feeding it to model. Predicted image
tensor is converted back to PIL image to display top 5 categories of flowers predicted. 
    




## Install



This project requires **Python** and the following Python libraries installed:



- [NumPy](http://www.numpy.org/)

- [torch](https://pytorch.org/)

- [matplotlib](http://matplotlib.org/)

- [torchvision](https://pytorch.org/docs/stable/torchvision/index.html)
- [PIL](https://pillow.readthedocs.io/en/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)




## Data



This 102 category dataset, consist of 102 flower categories. The flowers chosen to be flower commonly occuring in the United Kingdom. 
Each class consists of between 40 and 258 images. The images have large scale, pose and light variations. In addition, there are categories 
that have large variations within the category and several very similar categories. The dataset is visualized using isomap with shape and 
colour features.  Reference (http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

File cat_to_name.json is loaded to map index to categories of flowers. 


## Implementation 
Transferring learning approach is implemented to train model. Pretrained VGG16 model from Imagenet is used to implement tranfering learning.
Fully connected classifier of the model is designed with multiple layers, dropout and logSoftmax as output activation function. Testing 
accuracy of 0.816 is easily acheived after training for 5 epochs. Classifier part of the model is saved in checkpoint after training, 
later model is reloaded from checkpoint to make prediction on test images. PIL image processing library is used to convert raw image into
tensor and tensor back to raw image.       