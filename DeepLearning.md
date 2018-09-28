# Introduction to Deep Learning - Deep Learning basics with Python, TensorFlow and Keras p.1

Welcome everyone to an updated deep learning with Python and Tensorflow tutrial mini-series. 

Since doing the first deep learning with TensorFlow course a little over 2 years ago, much has changed. It's nowhere near as complicated to get started, nor do you need to know as much to be successful with deep learning.

If you're interested in more of the details with how TensorFlow works, youcan still chek the previous tutorials, as they go over the more raw TensorFlow. This is more of a deep learning quick start!

To begin, we nned to find some balance between neural newtworks a total black box, und understanding every single detail with them.

Let's show a typical model:

[image]

A basic neurla network consist of an input layer, which is just your data, in numerical form. After your input layer, you will have some number of what are called "hidden" layers. A hidden layer is just in between your input and output layers. One hidden layer means you just have a neural network. Two or more hidden layers? Boom, you've got a deep neual network!

Why is this? Well, if you just have a single hidden layer, the model is going to only learn linear relationships.  

If you have many hidden layers, you can begin to learn non-linear relationships between your input and output layers.

A single neuron might look as follows:
[Image]

So this is really where magic happens. The idea is a single neuron is just some of all the inputs x weights, fed through some sort of activation functions. The activation function is meant to simulate a neuron firing or not. A simple example would be a stepper function, where, at some point, the threshold is crossed, and the neuron fires a 1, else a 0. Let's say that neuron is in the first hidden layer, and it's going to communicate with the next hidden layer. So it's going to send it's 0 or 1 sigal, multiplied by the weights, to the next neuron, and this is the process for all neurons and all layers.  

The mathematical challange for the artificial neural newtwork is to best optimize thousands or millions or whatever number of weights you have, so that your output layer results in what you were hopping for. Solving for this problem, and building out the layers of our neural network model is exactly what TensorFlor is for.  
TensorFlow is used for all things "operations on tensors." A tensor in this case is nothing fancy. It's a multi-dimensional array.  

To install TensorFlow, simply do a:  
> pip install --upgrade tensorflow

Following the release of deep learning libraries, higher-level API-like libraries came out, which sit on top of the deep learning libraries, like TensorFlow, which make building, testing, and tweaking model even more simple. One such library that has easily become the most popular is Keras.

> import tensorflow.keras as keras

For this tutorial, I am going to be using TensorFlow version 1.10. You can figure out your version:  
> import tensorflow as ts  
> print(tf./__version/__)  
> 1.10.0  

Once we've got tensorflow imported, we can then begin to prepare our data, model it, and then train it. For the sake of simplicity, we'll be using the most common "hello world" example for deep learning, which is the minist dataset. It's a dataset of hand-written digits, 0 through 9. It's 28x28 iamges of these hand-written digits. We will show an example of using outside data das well, but, for now, let's load in this data:

> mnist = tf/keras.datasets.mnist
> (x_train, y_train), (x_test, y_test) = mnist.load_data()

When you're working with your own collected data, chances are, it won't be packaged up so nicely, and you'll spend a bit more time and effort on this step.  

What exactly do we have here? Let's take a quick peak.

So the x_train data is the "feature." In this case, the features are pixel values of the 28x28 images of these digits 0-9. The y_train is the label (is it a 0, 1, 2...9)

The testing variant of these variables is the "out of sample" examples that we will use. These are examples from our data that we're going to set aside, reserving them for testing the model.

Neural networks are exceptionally good at fitting to data, so much they will commonly over-fit the data. Our real hope is that the neural network doesn't just memorize our data and that it instead "generalizes" and learns the actual probelm and patterns associated with it.

Let's look at this actual data:
> print(x_train[0])



Alright, could we visualize this?


