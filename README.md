# Robot-Simulation-through-Gesture-Recognition
## Introduction

 

There have been many robotic applications where a smooth and real time control is needed according to human requirements. Sometimes it is very difficult to transfer input to the robot using the keyboard as the human needs to convert his input from ideas to keyboard values. It is tricky in the case of robot movement in different directions. 
Robot control through hand gestures is a developing area where a human can ask the robot to do something just by hand movements and no need to enter numeric values using the keyboard. Also, the input information could be transferred over the network to control a robot remotely.This can be used to control any type of robot for different actuation. 
It can be used to control the robotic arm remotely where the human presence is not conducive for his/her safety or to control a TV or PC monitor in order to replace the infrared remote.



![image](https://user-images.githubusercontent.com/60650532/125906505-3104899f-8737-48a0-93b9-a82d97b7f7dd.png)



## Aim and Objectives

So in our exploratory project we want to move a bot in a plane through our different hand gestures. For this we need to implement techniques such as deep learning for making a model that can classify different hand gestures.Also we need to apply opencv for live camera feed and taking images out of it to give final  command to the bot. <br/>
 
![image](https://user-images.githubusercontent.com/60650532/126115952-4e164d65-1a56-4069-b164-94e13e353828.png)


## Theory and Working

#### GESTURE RECOGNITION
Model is one of the most important part  of  our project as it will give the correct meaning of gesture like  what  gesture  is.CNN is the most acceptable technique for categorical image classification.   
#### Data Collection

First of all to train the model we need data that suit our requirement .Initially we surfed online but couldn't find a good dataset  so we decided to make our dataset with the help of opencv.We created 10 gestures and augmented multiple images of them. <br/>
![image](https://user-images.githubusercontent.com/60650532/126116019-c22e7bb6-725b-48c6-91cc-2ec465e0b813.png)




          




#### Loading Data and Model Training
##### Now loading this data to google colab for final training .
  ``` python 
 path_to_folders = "/content/train/"
im_folders = os.listdir('/content/train/')
images ={'image':[],'label':[]}
for i in im_folders:
    path = os.listdir(path_to_folders+i)
    for j in path:
          img = Image.open(path_to_folders+i+'/'+j)
          img = img.resize((150, 150))
          arr = np.array(img)
          images['image'].append(arr)
          images['label'].append(i)
x_data = np.array(images['image'],dtype='float32')
y_data = np.array(images['label'])
x_data = x_data/255
from sklearn.utils import shuffle
#x_data = x_data/255
x_data, y_data = shuffle(x_data,y_data)
le = LabelEncoder()
print(y_data.shape)
y_data = le.fit_transform(y_data)
num_classes = 10
y_data = to_categorical(y_data,num_classes)
x_data = x_data.reshape(-1,150,150,1)
 ```

 ##### MODEL ARCHITECTURE

![image](https://user-images.githubusercontent.com/60650532/126116204-d758a3f6-726c-4fd2-9d1c-1e42723923fb.png)

 


 ##### Final Training using Keras Library 
``` python
model.fit(x_train,y_train,shuffle=True,batch_size=32,epochs=15,validation_data=(x_validate,y_validate))
```
Final train accuracy - 99.92% <br/>
Final Validation Accuracy - 93% <br/>
 
Final Test Accuracy - 92.43% <br/>
 

###### Saving the model 
``` python
model.save('myt2.h5') 
from google.colab import files
files.download('myt2.h5')
```
## Importing Pybullet and Bot Arena

Now after saving the model we need a bot on which we can run commands and see the movement .We initially planned to implement it on a real robot using an arduino but it was not possible this time so we have to restrict ourselves to a  virtual robot that is built using pybullet. <br/>
You can install the following enviroment from [here](https://github.com/Robotics-Club-IIT-BHU/Pixelate_Main_Arena)

![image](https://user-images.githubusercontent.com/60650532/126116582-ba5f9897-d5e0-49c9-b303-23087f85834e.png)


```python
import gym
import pix_main_arena
import pybullet as p
env = gym.make("pix_main_arena-v0")
```


 



## Live input of Hand Gesture using opencv  
To  test and implement the simulation through gesture we requires to input our gesture to the system which is done through computer camera and its live feed is taken for continuous movement which can be done using opencv <br/>
```python
 cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        # Simulating mirror image
        frame = cv2.flip(frame, 1)
        
        #time.sleep(2)
        # Got this from collect-data.py
        # Coordinates of the ROI
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]
        
        # Resizing the ROI so it can be fed to the model for prediction
        roi = cv2.resize(roi, (150,150)) 
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)

  
```
![image](https://user-images.githubusercontent.com/60650532/126117329-ca4fa5a1-bcaa-437d-9d90-4bee63de8e78.png)


## Model loading and prediction
Now we load the saved model using keras library and feed the image taken above into the model to get predictions which basically output the gesture class. <br/>
```python
loaded_model = tf.keras.models.load_model('myt2.h5')`
result = loaded_model.predict(test2.reshape(1, 150,150, 1))`
```
 ![image](https://user-images.githubusercontent.com/60650532/126117403-be17762b-b5eb-4804-8d5f-d90ad3adb085.png)




## Bot Locomotion
To move bot in different direction that is to move it forward, backward,right and left we use the concept of differential motion in the following way-: <br/>
●	If both wheels on both right and left sides is given positive velocity the bot will move forward <br/>
●	If both wheels on both right and left sides is given negative velocity the bot will move backward <br/>
●	If both wheels on  right side is given  positive velocity and both wheels on  left side is given negative velocity bot will move leftward <br/>
●	If both wheels on right side is given  negative velocity and both wheels on  left side is given positive velocity bot will move rightward <br/>


  
In python code can be like this -:
```python
def movebackward():

        p.stepSimulation()
        
       env.move_husky(-2.,-2.5,-2.5,-2.5)`
            
def moveforward():

        p.stepSimulation()
        
       env.move_husky(8.5,8.5,8.5,8.5)
      
def moveleft():

        p.stepSimulation()
        
        env.move_husky(-5.9,4.9,-5.9,4.9)
        
def moveright():
       p.stepSimulation()
        env.move_husky(4.9,-5.9,4.9,-5.9)
 ```
        

So we now have to move the bot according to gestures. We selected four of our gestures mapping to movement in the following way - <br/>
* If gesture shows “5” then move the bot forward
* If gesture shows “2” then move the bot rightward
* If gesture shows “0” then move the bot leftward
* If gesture shows “3” then move the bot backward

![image](https://user-images.githubusercontent.com/60650532/126117458-1d83687e-0f59-45dd-918e-b3d855cab02d.png)

## Additional Feature

Apart from this main simulation we added an additional feature in which if there is an obstacle on a cell then even  if a  command is given to bot  to move towards that cell the bot will not move and will try to maintain distance from it <br/>

This is done by first locating the coordinates of the obstacles here the obstacles are created as black cells.So we find cells with block color using opencv techniques. <br/>
 



 

Then we use the aruco that is present on the bot to locate the bot position so if bot tries to move close any of the coordinates from obstacles list then we don't give the  command to bot even when the gesture is telling it to do so . <br/>
 

 
So before moving forward and backward we always check this that if it not moving close to any obstacle (black  cell). <br/>

●	The gesture recognition part of this project can be extended for many other purposes such as translator for mute people.It  has applications in virtual environment control, but also in sign language translation or musical creation . <br/>
![image](https://user-images.githubusercontent.com/60650532/126117510-c2b51c02-493b-4f4c-9d4f-ee7267c5300d.png)
###### Sample Video
![ezgif com-gif-maker](https://user-images.githubusercontent.com/60650532/126119103-3b5ee572-9542-46d7-9865-addd74112f9d.gif)


## Conclusion 

The need of telexistence in real-time inspired us to implement this system. A robot can be controlled over the network using hand gestures. The robot will move as per the gesture and would do movement and manipulation as per instruction. The proposed technique was tested in the environment which is shown above. Database of gestures was stored into the binary format of 150x150 pixels so it was taking less time and memory during pattern recognition. Due to cropped images of gestures, the system becomes more effective as the one image is sufficient for one type of gesture presentation. So we need not to store more than one image for the same gesture at different positions of the image.
 Experimental results show that the system detects hand gestures when the user stops moving hand for one second. The accuracy of the system is 95%. This method can be applied to any type of robot as Robot instructions were mapped on hand gesture signals. Currently, only four gestures are implemented for experimental purposes, which could be extended as per requirement. The system was implemented on a python idle with the help of an arena created using pybullet.
 The robot is also providing visual information which could be used for different purposes including surveillance and object manipulation. In future, we would like to build the whole system on a real robot, so that it would be much more real and deployable.

## FINAL NOTES

In the end, we would like to thank our supervisor Prof. M K Meshram for their valuable guidance and constant support. They helped us in completing this project even in the uncertain times of the Coronavirus Pandemic. We would also like to thank our parents and fellow batchmates for their constant support and encouragement when we felt struck while developing this project. <br/>

REFERENCES-:
https://www.sciencedirect.com/topics/computer-science/gesture-recognition#:~:text=Gesture%20recognition%20is%20an%20active,or%20musical%20creation%20%5B4%5D.
https://keras.io/api/layers/convolution_layers/
https://opencv.org/

