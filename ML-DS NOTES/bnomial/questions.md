# bnomial DS question of the day

## Tuesday, September 17th 2024

### Question:
__Brielle wants to build a machine learning model that will use traffic violations to predict how to distribute the city's police force.__

__She wants the model to predict the areas where new violations are likely to occur so the department can reinforce the security around those streets.__

__Which of the following is a potential problem that Brielle should consider?__

### Possible Asnwers:

    • There won't be any reliable way to evaluate this model. 
    
<details> <summary>Answer</summary><span style="color:red">INCORRECT</span></details>

    • The model may suffer from survivorship bias.

<details> <summary>Answer</summary><span style="color:red">INCORRECT</span></details>


    • The model may suffer from decline bias.

<details> <summary>Answer</summary><span style="color:red">INCORRECT</span></details>

    • The model may create a positive feedback loop.

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>

### Explanation:

Evaluating this model doesn't need to be complicated. Assuming that Brielle uses a Supervised Learning approach, she will have several options to assess the quality of the model predictions. Therefore, the first choice is incorrect.

[Survivorship bias](https://en.wikipedia.org/wiki/Survivorship_bias) is when we concentrate on samples that made it past a selection process and ignore those that did not. Nothing in the problem statement indicates that Brielle's model will suffer from this problem.

[Decline bias](https://en.wikipedia.org/wiki/Declinism) refers to the tendency to compare the past to the present, leading to the assumption that things are worse or becoming worse simply because change is occurring. The third choice is not a correct answer either.

Finally, this model may create a [positive feedback loop](https://en.wikipedia.org/wiki/Positive_feedback). The more you patrol a neighborhood, the more traffic violations you'll find. Communities with no police force will never report any violations, while heavily patrolled communities will have the lion's share of transgressions.

The model will use that data and make the problem worse: it will predict that new violations will happen in already problematic areas, sending more police to those communities at the expense of areas with lower reports. A few rounds of this, and you'll have most reports from a few places while violations are rampant everywhere.

### Recommended reading:

Check the description of a ["Positive Feedback Loop"](https://en.wikipedia.org/wiki/Positive_feedback) in Wikipedia.

["How Positive Feedback Loops Are Hurting AI Applications"](https://levelup.gitconnected.com/how-positive-feedback-loops-are-hurting-ai-applications-6eae0304521c) is an excellent article explaining the dangers of positive feedback loops in machine learning.

## Wednesday, September 18th 2024

### Question:
__Arianna is trying to learn how Convolutional Neural Networks work, so she decided to copy an online Keras example to start from somewhere.__

__Here is the core of the code she put together:__

``` 
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3),mmmm activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

```

__Based on the above code fragment, what are the correct statements regarding each layer's parameters (weights and biases)?__

### Possible Asnwers:


* The first convolutional layer has a total of 21,632 parameters.

<details> <summary>Answer</summary><span style="color:red">INCORRECT</span></details>

* The first max pooling layer has a total of 5,408 parameters.

<details> <summary>Answer</summary><span style="color:red">INCORRECT</span></details>

* The second convolutional layer has a total of 18,496 parameters.

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>

* The fully-connected layer has a total of 1,600 parameters.

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>

### Explanation:

We can compute the number of parameters of a convolutional layer using the following formula:

```
parameters = k * (f * h * w + 1)
```

Where k corresponds to the number of output filters from this layer, f corresponds to the number of filters coming from the previous layer, h corresponds to the kernel height and w to the kernel width. The value 1 corresponds to the bias parameter related to each filter. Here is the complete calculation for the first convolutional layer:

```
parameters = k * (f * h * w + 1)
parameters = 32 * (1 * 3 * 3 + 1) = 320
```

Max pooling layers don't have any parameters because they don't learn anything. The input to the first max pooling layer is 26x26x32, but the layer doesn't have any weights or biases associated with it.

The second convolutional layer does have 18,496 parameters. Let's check:

```
parameters = k * (f * h * w + 1)
parameters = 64 * (32 * 3 * 3 + 1) = 18,496
```

Finally, the fully-connected layer has 1,600 parameters. To compute this, we need to calculate the size of each layer to understand the input to the fully-connected layer:

* Input: 28x28x1 = 784
* Conv2D: 26x26x32 = 21,632
* MaxPool2D: 13x13x32 = 5,408
* Conv2D: 11x11x64 = 7,744
* MaxPool2D: 5x5x64 = 1,600

### Recommended reading:

Notice that the above values differ from the number of learnable parameters of each layer, but they are essential to understanding the size of the input to the fully-connected layer.
Recommended reading

Check ["Understanding and Calculating the number of Parameters in Convolution Neural Networks (CNNs)"](https://towardsdatascience.com/understanding-and-calculating-the-number-of-parameters-in-convolution-neural-networks-cnns-fc88790d530d) for instructions on how to compute the number of learnable parameters.

["Simple MNIST convnet"](https://keras.io/examples/vision/mnist_convnet/) is a Keras example showing this particular code fragment.

## Thursday, September 19th 2024

### Question:

__Esther had an excellent model already, but she had the budget to experiment a bit more and improve its results.__

__She was building a deep network to classify pictures. From the beginning, her Achilles' heel has been the size of her dataset. One of her teammates recommended she use a few data augmentation techniques.__

__Esther was all-in. Although she wasn't sure about the advantages of data augmentation, she was willing to do some research and start using it.__

__Which of the following statements about data augmentation are true?__


### Possible Asnwers:

* Esther can use data augmentation to expand her training dataset and assist her model in extracting and learning features regardless of their position, size, rotation, etc.

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>

* Esther can use data augmentation to expand the test dataset, have the model predict the original image plus each copy, and return an ensemble of those predictions.

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>

* Esther will benefit from the ability of data augmentation to act as a regularizer and help reduce overfitting.

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>

* Esther has to be careful because data augmentation will reduce the ability of her model to generalize to unseen images.

<details> <summary>Answer</summary><span style="color:red">INCORRECT</span></details>

### Explanation:

One significant advantage of data augmentation is its ability to make a model resilient to variations in the data. For example, assuming we are working with images, we can use data augmentation to generate synthetic copies of each picture and help the model learn features regardless of where and how they appear.

A few popular augmentation techniques when working with images are small rotations, horizontal and vertical flipping, turning the picture to grayscale, or cropping the image at different scales. The [following example](https://www.v7labs.com/blog/data-augmentation-guide) shows four versions of an image generated by changing the original picture's brightness, contrast, saturation, and hue:

<img src="images/augmenting_the_dataset.png">

Data augmentation is also helpful during testing time: Test-Time Augmentation is a technique where we augment samples before running them through a model, then average the prediction results. Test-Time Augmentation often results in better predictive performance.

Instead of predicting an individual sample from the test set, we can augment it and run each copy through the model. Esther is working on a classification problem, so her model will output a softmax vector for each sample. She can then average all these vectors and use the result to choose the correct class representing the original sample.

Using data augmentation, Esther can reduce overfitting and help her model perform better on unseen data. Data augmentation has a regularization effect. Increasing the training data through data augmentation decreases the model's variance and, in turn, increases the model's generalization ability. Therefore, the third choice is correct, but the fourth one is not.

### Recommended reading:

["The Essential Guide to Data Augmentation in Deep Learning"](https://www.v7labs.com/blog/data-augmentation-guide) is an excellent article discussing data augmentation in detail.

Check ["Test-Time augmentation"](https://articles.bnomial.com/test-time-augmentation) for an introduction that will help you make better predictions with your machine learning model.

## Friday, September 20th 2024

### Question:

__Serena has been given an intriguing project at her workplace. She has to design an object detection model that identifies different types of fruits in images taken from a grocery store.__

__Her goal is to create a model that can be easily modified and used for other similar tasks in the future.__
__To ensure that her model performs at its best, Serena needs a method to evaluate it effectively. This evaluation method should allow her to compare and choose the best among different versions of the model.__

__Which evaluation metrics should Serena use to evaluate her model?__

### Possible Asnwers:

* F1 score

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>

* Mean Average Precision (mAP)

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>

* Precision-Recall Curve

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>

* ROC Curve

<details> <summary>Answer</summary><span style="color:red">INCORRECT</span></details>

### Explanation:

The recall is useful for object detection, but it can't provide the full picture unless combined with Precision. High recall and low precision could lead to a model that is not useful. Therefore, Serena cannot rely solely on Recall as her key evaluation metric.

[ROC Curves](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=es-419) are not typically used for object detection tasks,
as there's no real concept of True Negatives, which are required to compute the False Positive Rate, one of the axes of the ROC curve. In object detection tasks, the number of bounding boxes that do not contain an object of interest is generally too large to handle effectively.

Instead, Serena could compute a [Precision-Recall Curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html). This curve is similar to the ROC curve but uses the model's precision instead of False Positive Rate, thereby avoiding the problem of True Negatives.

[Mean Average Precision (mAP)](https://medium.com/@timothycarlen/understanding-the-map-evaluation-metric-for-object-detection-a07fe6962cf3) is commonly used in object detection tasks to evaluate the overall performance of a model across all classes. It considers precision and recall and averages them over different Intersection over Union (IoU) thresholds, providing a single scalar value that Serena can use to compare different models.

Lastly, the [F1-score](https://en.wikipedia.org/wiki/F-score) is a good choice, as it considers both the precision and recall of the model, offering a balanced view of the model's performance.

### Recommended reading:

Check ["Classification: ROC Curve and AUC"](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=es-419) for an explanation of how to create and interpret a ROC curve. 

For more information about Precision-Recall curves, check [Scikit-Learn's documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html).

## Saturday, September 21st 2024

### Question:

__Katherine wants to use an ensemble model to process her dataset.  There's only one question for her to answer: Should she use bagging or boosting?__ 

__Both techniques have different advantages and disadvantages, and Katherina wants to ensure she evaluates them correctly before committing to one solution.__

__Both techniques have different advantages and disadvantages, and Katherina wants to ensure she evaluates them correctly before committing to one solution.__

__Which of the following statements are true about bagging and boosting?__

### Possible Answers:

* Bagging trains individual models sequentially, using the
results from the previous model to inform the selection of
training samples.

<details> <summary>Answer</summary><span style="color:red">INCORRECT</span></details>

* Boosting trains individual models sequentially, using the
results from the previous model to inform the selection of
training samples.

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>


* Bagging trains a group of models, each using a subset of
data selected randomly with replacement from the original
dataset.

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>

* Each model receives equal weight in bagging to compute the
final prediction, while boosting uses some way of weighing
each model.

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>

### Explanation:

Ensembling is where we combine a group of models to produce a
new model that yields better results than any initial
individual models. Bagging and boosting are two popular
ensemble techniques.

Bagging trains a group of models in parallel and independently
from each other. Each model uses a subset of the data randomly
selected with replacement from the original dataset. In
contrast, Boosting trains a group of learners sequentially,
using the results from each model to inform which samples to
use to train the next model.

This summary helps us conclude that the first choice is
incorrect, but the second and third choices are correct.

Finally, when computing the final prediction, bagging averages
out the results of each model. Boosting, however, weights each
model depending on its performance. Therefore, the fourth
choice is also correct.

### Recommended reading:

Check [Bagging vs. Boosting in Machine Learning: Difference
Between Bagging and Boosting](https://www.upgrad.com/blog/bagging-vs-boosting/) for a detailed comparison
between both techniques.

[What is the difference between Bagging and Boosting?](https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/) is
another great summary of both techniques and their
advantages and disadvantages.

## Sunday, 22nd September 2024

### Question:

__Sienna realized she needed more than convolutional layers to process her image dataset.__

__After stacking a few convolutional layers, her model started to make progress. Unfortunately, only very similar images returned positive results. Sienna discovered that her model lacked translation invariance: it was paying too much attention to the precise location of every feature.__

__Fortunately, Sienna found out that she could use pooling layers.__

__Which of the following statements about pooling layers are correct?__

### Possible Answers:

* During the training process, the network will learn the best configuration for the pooling layer.

<details> <summary>Answer</summary><span style="color:grey">N/A</span></details>

* A pooling layer with a stride of 2 will cut the number of
feature maps from the previous convolutional layer in
half.

<details> <summary>Answer</summary><span style="color:red">INCORRECT</span></details>

* Pooling layers create the same number of pooled feature
maps.

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>


* Average pooling and max pooling are two of the most common
pooling operations.

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>

### Explanation:

Pooling layers don't have any learnable parameters. When
designing the model, Sienna must specify the pooling operation
and configuration she wants to use.

Pooling layers work on each feature map independently and,
depending on the pool size and stride, downsample these feature
maps. The result is always a new set of pooled feature maps.
Therefore, the second choice is incorrect, but the third is
correct.

Finally, Max Pooling and Average Pooling are the two most
common pooling operations. Average Pooling computes the average
value of each patch, while Max Pooling calculates the maximum
value.

### Recommended reading:

["Max Pooling in Convolutional Neural Network and Its
Features"](https://analyticsindiamag.com/ai-mysteries/max-pooling-in-convolutional-neural-network-and-its-features/) is a great introduction to Max Pooling.

Check ["A Gentle Introduction to Pooling Layers for
Convolutional Neural Networks"](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/) for more information about
how pooling layers work.

## Monday, 23rd September 2024

### Question:

__Juniper learned that designing a neural network architecture for a supervised classification problem wasn't hard.__

__Although most of it needed experimentation, one thing Juniper could count on was the design of the output layer of the network.__

__Which of the following is a correct statement about the neurons in the output layer of a classification network?__

### Possible Answers:

* The number of neurons in the output layer should always match the number of classes.

<details> <summary>Answer</summary><span style="color:grey">N/A</span></details>

* The number of neurons in the output layer doesn't necessarily need to match the number of classes.

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>

* The number of neurons in the output layer should always be greater than one.

<details> <summary>Answer</summary><span style="color:red">INCORRECT</span></details>


* The number of neurons in the output layer should always be a multiple of 2.

<details> <summary>Answer</summary><span style="color:grey">N/A</span></details>


### Explanation:

When working on a multi-class classification problem, setting the output layer to the same number of classes we are interested in predicting is common. But what happens when we are working on a binary classification problem?

A network to solve a binary classification problem doesn't need an output with two neurons. Instead, we can use a single neuron to determine the class by deciding on a cutoff threshold. For example, the result could be positive if the output exceeds 0.5 and negative otherwise.

That means we can solve a problem requiring two classes with a single neuron.

We can stretch the same idea to multi-class classification problems: We could interpret the output layer as a binary result, allowing us to represent multiple classes with fewer neurons. For example, we would need only two neurons to classify instances into four different categories (00, 01, 10, 11.) This approach, although not popular, it's possible.

Therefore, the second choice is the correct answer to this question.

### Recommended Reading:


["But what is a neural network?"](https://www.youtube.com/watch?v=aircAruvnKk) is Grant Sanderson's introduction to neural networks on YouTube. Highly recommended!

[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) is a free online book written by [Michael Nielsen](https://x.com/michael_nielsen).
