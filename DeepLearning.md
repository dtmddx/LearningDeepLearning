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
> import matplotlib.pyplot as plt
> plt.imshow(x_train[0], cmap=plt.cm.binary)
> plt.show()

Okay, that makes sense. How about the value for y_train with the same index?

> print(y_train[0])
> 5

It's generally a good idea to "normalize" your data. This typically involves scaling the data to be between 0 and 1, or maybe -1 and positive 1. In our case, each "pixel" is a feature, and each feature currently ranges from 0 to 255. Not quite 0 to 1. Let's change that with a handy utility function:

> x_train = tf.keras.utils.normalize(x_train, axis=1)
> x_test = tf.keras.utils.normalize(x_test, axis=1)

Let's peak one mor time:
> print(x_train[0])
> plt.imshow(x_train[0], cmap=plt.cm.binary)
> plt.show()

Alright, still a 5. Now let's build our model!

> model = tf.keras.models.Sequential() # Sequential 序列
A sequential model is what you're going to use most of the time. It just means things are going to go in direct order. A feed forward model. No going backwards...for now.

Now, we'll pop in layers. Recall our neural network images? Was the input layer flat, or was it multi-dimensional? It was flat. So, we need to take this 28x28 image, and make it a flat 1x784. There are many ways for use to do this, but keras has a Flatten layer built just for us, so we'll use that.

> model.add(tf.keras.layers.Flatten())

This will serve as our input layer. It's going to take the data we throw at it, and just flatten it for us. Next, we want our hidden layers. We're going to go with the simplest neural network layer, which is just **dense** layer. This refers to the fact that it's a densely-connected layer, meaning  it's "fully connected," where each node connects to each prior and subsequent node. Just like our image.

> model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

This layer hase 128 units. The activation function is relu, short for rectified linear. Currently, relu is the activation function you should just default to. There are many more to test for sure, but, if you don't know what tu use, use relu to start.

Let's add another identical layer for good measure.
> model.add(tf.keras.layer.Dense(128, activation=tf.nn.relu))

Now, we're ready for an output layer.
> model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

This is our final layer. It has 10 nodes. 1 node per possible number prediction. In this case, our activation function is a softmax function, since we're really actually looking for something more like a probability distribution of which of the possible prediction options this thing we're passing features through of is. Great, our model is done.

Now, we need to "compile" the model. This is where we pass the setting for actually optimizing/training the model we've defined.

> model.compile(optimize='adam',
		loss='sparse_categorical_crossentropy',
		matrics=['accuracy'])

Remember why we picked relu as an activation function? Same thing is true for the Adam optimizer. It's just a great default to start with.

Next, we have our loss metric. Loss is a calculation of error. A neural network doesn't actually attempt to maximize accuarcy. It attempts to minimize loss. Again, there are many choices, but some form of categorical crossentropy is a good start for a classificaton task like this.

Now, we fit!
> model.fit(x_train, y_train, epochs=3)

As we train, we can see loss goes down, and accuracy improves quite quickly to 98~99%.

Now that's loss and accuracy for in-sample data. Getting a high accuracy and low loss might mean your model learned how to classify digis in general or it simply memorized every single example you showed it. This is why we need to test on out-of-sample data (data we didn't use to train the model).


> val_loss, val_acc = model.evaluate(x_test, y_test)
> print(val_loss)
> print(val_acc)

It's going to be very likely your accuracy out of sample is a bit worse, same with loss. In fact, it should be a red flag if it's identical, or better.

Finally, with your model, you can save it easily:
> model.save('epic_num_reader.model')

Load it back:
new_model = tf.keras.models.load_model('epic_num_reader.model')

Finally, make predictions!

> predictions = new_model.predict(x_test)
> print(predictions)

Awesome! Okay, I think that covers all of the "quick start" types of things with keras. this is just barely scratching the surface of what's available to you, so start poking around Tensorflow and Keras documentation.

Also check you the Machine Learning and Learn Machine Learning subreddits to stay up to date on news and information surrounding deep learning.

If you have further questions too, you can join our Pyhong Dicord. Til next time.


# Loading in your own data - Deep Learning basics with Python, TensorFlow and Keras p.2

Welcome to a tutorial where we'll be discussing how to loading in our own outside datasets, which comes with all sorts of challanges!

First, we need a dataset. Let's grab the Dog vs Cats dataset from Microsoft. If this dataset disappears, someone let me know. I will host it myself.

Now that you have the dataset, it's currently compressed. Upzip the dataset, and you should find that it creates a directory calledn PetImages. Inside of that, we have Cat and Dog directories, which are then filled with images of cats and dogs. Easy enough! Let's play with this dataset! First, we need to understand how we will convert this dataset to training data. We have a few issues right out of the gate. The largest issue is not all of these images are the same size. While we can eventually have variable-sized layers in neural networks, this is not the most basic thing to archieve. We're going to want to reshape things for now so every image has the same dimensions. Next, we may or may not want to keep color. To begin, install matplotlib if you don't already have it, as well as opencv.

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "your directory where cat and dog pics stored."

CATEGORIES = ['Dog', 'Cat']

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path)
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
	plt.imshow(img_array, cmap='gray')
	plt.show()

	break
    break



you will see a dog.

print(img_array)

and the next is the shape of array.
print(img_array.shape)

So that's a 375 tall, 500 wide, and 3-channel image. 3-channel is because it's RGB(color). We definitely don't want the images that big, but also various images are different shapes, and this is also a problem.

IMG_SIZE = 50

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

Hmm, that's a bit blurry I'd say. Let's go with 100x100?

Better. Let's try that. Next, we're going to want to create training data and all that, but, first, we should set aside some images for final testing. I am going to just manually create a directory called Testing, and then create 2 directories inside of there, one for Dog and one for Cat. From here, I am just going to move the first 15 images from both Dog and Cat into the training versions. Make sure you move them, not copy. We will use this for our final tests.

Now, we want to begin building our training data!

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
	class_num = CATEGORIES.index(category)

	for img in tqdm(os.listdir(path)):
	    try:
	        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
		new_array = cr2.resize(img_array, (IMG_SIZE, IMG_SIZE))
		traning_data.append([new_array, class_num])
	    except Exception as e:
	    	pass

create_training_data()

print(len(training_data))

Great, we have almost 25k samples! That's awesome.

On thin we want to do is make sure our data is balanced. In the case of this dataset, I can see that the dataset started off as being balanced. 

By balanced, I mean there are the same number of examples for each class (same number of dogs and cats). If not balanced, you either wanto pass the class weights to the model, so that that is  can measure error appropriately, or balace your samples by trimming the larger set to be the same sas as the smaller set.

f you do not balance, the model will initially learn that the best thing to do is predict only one class, whichever is the most common. Then, it will often get struk here. In our case though, this data is already balanced, so that's easy enough. Maybe later we'll have a dataset that isn't balanced so nicely.

Also, if you have a dataset that is too large to fit into your ram, you can batch-load in your data. There are many ways to do this, some outside of TensorFlor and some build in. We may discuss this further, but, for now, we're mainly tring to cover how your data should look, be shaped, and fed into the models.

Next, we want to shuffle the data. Right now our data is just all dogs, then all cats. This will usually wind up causing trouble too, as, initially, the classifier will learn to just predict dogs always. Then it will shift to oh, just predict all cats! Going back and forth like this is no good either.

> import random
> random.shuffle(training_data)
Our training_data is a list, meaning it's nutable, so it's now nicely shuffled. We can confirm this by iterating over a few of the inital samples and printing out the class.

for sample in training_data[:10]:
    print(sample[1])


Great! We've got the classes nicely mixed in! Time to make our model!
X = []
y = []

for features, label in traning_data:
    X.append(features)
    y.append(label)


print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))




# Convolutional Neural Networks - Deep Learning basics with Python, TensorFlow and Keras p.3
Welcome to a tutorial where we'll be discussing Convolutional Neural Networks (Convnets and CNN), using one to classify dogs and cats with the dataset we built in the previous tutorial.

The Convolutional Neural Network gained popularity through its use with image data, and is currently the state of the art for detecting what an image it, or what is contained in the image.

The basic CNN structure is a follows: Convolution -> Pooling -> Convolution -> Pooling -> Fully Connected Layer -> Output

Convolution is the act of taking the original data, and creating feature maps from it. Pooling is down-sampling, most often in the form of "max-pooling," where we select a region, and then take the maximun value in that region, and that becomes the new value for entire region. Fully Connected Layers are typical neural networks, where all nodes are "fully connected." The convolutional layers are not fully connected like a traditional neural network.

Okay, so now let's depict what's happening. We'll start with an image of cat:
 [image]
 Then "conver to pixels"

 [image]

 For the purposes of this tutorial, assume each square is a pixel. Next, for the vonvolution step, we're going to take a certain window, and find features within that window.

 That windows's features are now just a single pixel-sized feature in new featuremap, but we will have multiple layers of featuremaps in reality.

Next, we slide that window over and continue the process. There will be some overlap, you can determine how much you want, you just do not want to be skipping an pixels, of course.

[image]
Now you continue this process until you've covered the entire image, and then you will have a fearuemap. Typically the fearuemap is just more pixel values, just a very simplified one:


