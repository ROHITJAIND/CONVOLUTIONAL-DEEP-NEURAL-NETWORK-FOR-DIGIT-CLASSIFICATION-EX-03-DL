# EX-03 Convolutional Deep Neural Network for Digit Classification

### Aim:
To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

### Problem Statement and Dataset

- Digit classification and to verify the response for scanned handwritten images.
- The MNIST dataset is a collection of handwritten digits.
- The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.
- The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

### Neural Network Model

<img height=25% src="https://github.com/user-attachments/assets/09c700c3-1ac5-44c2-aff2-6bec1e4e0478">

### DESIGN STEPS
- **Step 1:** Import tensorflow and preprocessing libraries
- **Step 2:** Download and load the dataset
- **Step 3:** Scale the dataset between it's min and max values
- **Step 4:** Using one hot encode, encode the categorical values
- **Step 5:** Split the data into train and test
- **Step 6:** Build the convolutional neural network model
- **Step 7:** Train the model with the training data
- **Step 8:** Plot the performance plot
- **Step 9:** Evaluate the model with the testing data
- **Step 10:** Fit the model and predict the single input
### Program:
```Python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(xtrain, ytrain),(xtest, ytest)=mnist.load_data()
xtrainS=xtrain/255.0
xtestS=xtest/255.0
ytren = utils.to_categorical(ytrain,10)
yteen = utils.to_categorical(ytest,10)
xtrainS = xtrainS.reshape(-1,28,28,1)
xtestS = xtestS.reshape(-1,28,28,1)
model = keras.Sequential()
model.add(layers.Input (shape=(28,28,1)))
model.add(layers.Conv2D (filters=32, kernel_size=(7,7), activation='relu'))
model.add(layers.MaxPool2D (pool_size=(3,3)))
model.add(layers.Flatten())
model.add(layers.Dense (32, activation='relu'))
model.add(layers.Dense (16, activation='relu'))
model.add(layers.Dense (8, activation='relu'))
model.add(layers.Dense (10, activation='softmax'))
model.summary()
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(xtrainS,ytren,epochs=15,batch_size=256,validation_data=(xtestS,yteen))
metrics = pd.DataFrame(model.history.history)
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
ypred = np.argmax(model.predict(xtestS), axis=1)
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))
img = image.load_img('image.png')
tensor_img = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(tensor_img,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
plt.title('ROHIT JAIN D 212222230120')
np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)
```
### Output:

### Training Loss, Validation Loss Vs Iteration Plot

<img height=15% width=48% src="https://github.com/user-attachments/assets/5431bfaa-1667-4f18-a6b3-6c2edddbadee"><img height=15% width=48% src="https://github.com/user-attachments/assets/52a0cef9-d003-45d6-8fc8-191b2a860242">


<table>
<tr>
<td width=48%>
  
### Classification Report
![image](https://github.com/user-attachments/assets/8cd77ce6-26a2-48da-ac20-f9f637518bf3)
</td> 
<td valign=top>

### Confusion Matrix
![image](https://github.com/user-attachments/assets/51f3cea9-2d97-49bb-9423-c4103ec16373)</td>
</tr> 
</table>


### New Sample Data Prediction

<img height=20% src="https://github.com/user-attachments/assets/454f1f51-75a3-4be5-9ab8-5e65a628b160">


### RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
