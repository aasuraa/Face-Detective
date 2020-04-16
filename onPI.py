"""
    Final Year Project
    Face detection, identification, and text alert with raspberry pi and external webcam
    
    @author: Sagar Ghimire
    @status: INCOMPLETE
"""

from PIL import Image, ImageTk
import tkinter as tk
import threading
import cv2
import os
import pickle
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pylab as plt
import random
import boto3
import RPi.GPIO as GPIO
from datetime import datetime
import time
from botocore.exceptions import NoCredentialsError

# face detection cascade
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

class Application:
    def __init__(self, dataPath = "./output/"):
        """
            Initialize application GUI which uses OpenCV + Tkinter. It displays video frame and buttons.
        """
        # initialize webcam capture
        self.vs = cv2.VideoCapture(0)
        self.fLen = self.vs.get(3)
        self.fWid = self.vs.get(4)
        
        self.i = 10.0                                                   # initial horizontal servo position
        self.iVer = 1.5                                                 # initial vertical servo position
        self.horzFlag = True                                            # initial servo movement direction
                
        self.position(self.i, self.iVer)                                # initial position

        self.dataPath = dataPath                                        # store output path
        self.dataPathImg = None                                         # path to store the images
        self.current_image = None                                       # current image from the camera
        self.lastFace = None                                            # last face seen on the camera
        self.count = 0                                                  # counter used for adding user
        self.uname = None                                               # variable for user name
        self.rec = False                                                # flag to use recognizer or not
        self.targets = self.getTargets()                                # list of target users
        self.now = abs(datetime.now().minute-5)                         # current time
        
        # TODO: amazon access key                                       # required amazon access key for text alert
        self.ACCESS_KEY = "" 
        self.SECRET_KEY = ""

        # loading trained models for recognizion
        self.model = tf.keras.models.load_model(self.dataPath+"newModel.h5")    
        self.lb = self.load(self.dataPath+"lb")
        print("[INFO] models loaded...")

        self.root = tk.Tk()                                             # initialize root window
        self.root.title("Face Detective")                               # set window title

        # self.destructor function gets fired when the window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        
        frame1 = tk.Frame(self.root)                                    # main frame of GUI
        frame1.pack(side="top")
        self.panel = tk.Label(frame1)                                   # initialize video panel
        self.panel.pack(padx=10, pady=10, side="top")

        # Buttons on frame 1
        nameLabel = tk.Label(frame1, text='User Name:')
        nameLabel.pack(fill="none", side="left", expand=False, padx=10, pady=10)
        self.nameEntry = tk.Entry(frame1)
        self.nameEntry.pack(fill="none", side="left", expand=True, padx=10, pady=10)
        targetbtn = tk.Button(frame1, text="Target User", command=self.targetUser)
        targetbtn.pack(fill="none", side="right", expand=True, padx=10, pady=10)
        targetbtn = tk.Button(frame1, text="Remove Target", command=self.targetRemove)
        targetbtn.pack(fill="none", side="right", expand=True, padx=10, pady=10)
        rmvbtn = tk.Button(frame1, text="Remove User", command=self.removeUser)
        rmvbtn.pack(fill="none", side="right", expand=True, padx=10, pady=10)
        btn1 = tk.Button(frame1, text="Add User", command=self.addUser)
        btn1.pack(fill="none", side="right", expand=True, padx=10, pady=10)

        frame2 = tk.Frame(self.root)
        frame2.pack(side="top")

        # Buttons on frame 2
        btn2 = tk.Button(frame2, text="Train Recognizer", command=self.trainNeural)
        btn2.pack(fill="none", side="left", expand=True, padx=10, pady=10)
        btn3 = tk.Button(frame2, text="Toggel Recognizer", command=self.toggelRec)
        btn3.pack(fill="none", side="left", expand=True, padx=10, pady=10)
        btn4 = tk.Button(frame2, text="Exit", command=self.destructor)
        btn4.pack(fill="none", side="left", expand=True, padx=10, pady=10)

        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()
        
    def horizontalScan(self):
        """
        Scan for a face moving the servos horizontally left to right. Stops when a face is detected.
        """
        if self.iVer != 1.5:
            self.iVer = 1.5
            self.verPosition(self.iVer)
        servoPIN = 22
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(servoPIN, GPIO.OUT)
        p = GPIO.PWM(servoPIN, 50)
        p.start(self.i)
        
        if self.horzFlag == True:
            self.i += .2
            p.ChangeDutyCycle(self.i)
            time.sleep(.1)
            if self.i > 11.0:
                self.horzFlag = False
        else:
            self.i -= .2
            p.ChangeDutyCycle(self.i)
            time.sleep(.1)
            if self.i < 2.0:
                self.horzFlag = True
    
        p.stop()
        GPIO.cleanup()

    def videoLoop(self):
        """
            Get frame from the video stream, detect a face, and show it in tkinter GUI
        """
        ref, frame = self.vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # detect face and draw rectangel around it
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # check if the face is present inside the 90% of the inner frame
            if((x < 0.1*self.fLen) or (y < 0.1*self.fWid) or (x+w > 0.9*self.fLen) or (y+h > 0.9*self.fWid)):
                self.mapServoFace(x, y, w, h)
            self.lastFace = gray[y:y+h, x:x+w]                          # saving the last face seen
        
        if len(faces) == 0:                                             # no face, move horizontally
            self.horizontalScan()

        if self.count <= 100 and self.count != 0:                       # adding a face using a counter
            self.addUser()
        if self.count > 100:
            print("[INFO] Face Added to folder")
            self.count = 0
            self.nameEntry.delete(0, 'end')

        if ref:                                                         # loding frame to tkinter GUI
            frame = cv2.flip(frame, 1)
            if self.rec == True:                                        # started to recognize, flag is ON
                image = cv2.resize(cv2.equalizeHist(self.lastFace), (30, 30))
                image = image.astype("float") / 255.0
                image = image.reshape(1, *image.shape, 1)
                preds = self.model.predict(image)
                i = preds.argmax(axis=1)[0]
                label = self.lb.classes_[i]
                
                if (abs(self.now - datetime.now().minute) >= 1):        # check if within a minute interval and send the text
                    if label in self.targets:                           # check if in target
                        cv2.imwrite("./output/lastFace.jpg", self.lastFace)
                        print("Sending Text Alert" + label + "seen")
                        #self.sendText(label)                           # this sends the text
                        self.now = datetime.now().minute

                # draw the class label + probability on the output image
                text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, True)

            # displaying in the GUI
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=self.current_image)        # convert image for tkinter
            self.panel.imgtk = imgtk                                    # anchor imgtk so it does not be deleted by garbage-collector
            self.panel.config(image=imgtk)                              # show the image
        self.root.after(3, self.videoLoop)                              # call the same function after 3 milliseconds

    def sendText(self, pName):
        """
            Takes the name of the user found, calls upload function to upload into s3 bucket, and sends a link to the user.    
        """
        faceURL = "https://facedetective-2020.s3-us-west-2.amazonaws.com/lastFace.jpg"

        uploaded = self.upload_to_aws("./output/lastFace.jpg", "facedetective-2020", "lastFace.jpg")
        if uploaded:                                                    # checks if upload is a success
            client = boto3.client(
                        "sns",
                        aws_access_key_id=self.ACCESS_KEY,
                        aws_secret_access_key=self.SECRET_KEY,
                        region_name="us-east-1"
                    )
            client.publish(
                        PhoneNumber="+17814924960",
                        # PhoneNumber="+61432848561",
                        Message="ALERT! "+ pName +" seen.. "+ faceURL
                    )
            print("[MSG] Text sent...")
            self.now = datetime.now().minute

    def upload_to_aws(self, local_file, bucket, s3_file):
        """
            Takes in file address, bucket, and file name for upload.
            returns true or false
        """
        s3 = boto3.client('s3', aws_access_key_id= self.ACCESS_KEY,
                          aws_secret_access_key= self.SECRET_KEY)

        try:
            s3.upload_file(local_file, bucket, s3_file)
            print("Upload Successful")
            return True
        except FileNotFoundError:
            print("The file was not found")
            return False
        except NoCredentialsError:
            print("Credentials not available")
            return False

    def toggelRec(self):
        """
            Toggel for recognizer to improve the performance.
        """
        if self.rec == True:
            self.rec = False
        else:
            self.rec = True

    def targetUser(self):
        """
            Write, and close the txt file containing targets
        """
        if not self.nameEntry.get():        # if not True i.e empty
            print("[INFO] Empty User Name!")
        else:
            f = open("./output/targets.txt", "w+")
            f.write(self.nameEntry.get()+"\r")
            f.close()
            print("[INFO] target set...")
            self.nameEntry.delete(0, 'end')


    def targetRemove(self):
        """
            Read, write, and close the txt file containing targets
        """
        if not self.nameEntry.get():
            print("[INFO] Empty User Name!")
        else:
            print("[INFO] removing target...")
            with open("./output/targets.txt", "r") as f:
                if f.mode == "r":
                    lines = f.readlines()
                    f.close()
                    with open("./output/targets.txt", "w+") as f:
                        for line in lines:
                            if line.strip("\n") != self.nameEntry.get():
                                f.write(line)
                        print("[INFO] target removed...")
                        f.close()
            self.nameEntry.delete(0, 'end')

    def getTargets(self):
        """
            Runs at the beginning of the application to get targets
        """
        tars = ["nuser"]
        with open("./output/targets.txt", "r") as f:
            if f.mode == "r":
                lines = f.readlines()
            for line in lines:
                tars.append(line.strip("\n"))
            return tars

    def removeUser(self):
        pass

    def load_images(self, path):
        """
            Loads and initialize the data and labels form the path of the data set given
            :param path: Path to user folders
            :return: array of images and labels respectively
                labels meaning categorical name
        """

        print("[INFO] loading images...")
        data = []
        labels = []

        # grab the image paths and randomly shuffle them
        imagePaths = os.listdir(path)                                   # list of folders in the dataset
        numCal = len(imagePaths)
        random.seed(42)
        random.shuffle(imagePaths)
                
        for imagePath in imagePaths:                                    # loop over the input images
            label = imagePath
            imagePath = path+imagePath+'/'
            for img in os.listdir(imagePath):
                image = cv2.imread(imagePath+img, 0) / 255.0
                image = image.reshape( *image.shape, 1)                 # has to reshape image for keras NN training
                data.append(image)                                      # already resized to 30x30 while capturing data
                labels.append(label)
        return data, labels, numCal

    def trainNeural(self):
        """
            takes the pictures taken and use it to train the neural network architecture model
            saves the trained model
        """
        print("[INFO] initializing train...")
        # load images for training
        path = "./dataset/"
        data, labels, numCal = self.load_images(path)
        
        # splitting data into train and test size, 80-20
        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, random_state=42)

        print("[INFO] data split...")
        # change to arrays instead of using list for input, keras requirement
        trainX = np.array(trainX)
        testX = np.array(testX)
        trainY = np.array(trainY)
        testY = np.array(testY)

        print(len(trainX)+len(testX), len(trainX), len(testX), len(trainY), len(testX), trainX.shape)

        # one hot encoding for labels
        lb = LabelBinarizer()
        trainY = lb.fit_transform(trainY)                               # fit finds all unique class labels
        testY = lb.transform(testY)                                     # no fit needed as class labels already found

        # model architecture code
        print("[INFO] building architecture...")
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(1024, input_shape=(30, 30, 1), name='inputLayer', activation="relu"))
        model.add(tf.keras.layers.Dense(512, name='hiddenLayer1', activation="relu"))
        # model.add(tf.keras.layers.Dense(256, name='hiddenLayer2', activation="relu"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(len(lb.classes_), name='outputLayer', activation="softmax"))

        model.summary()                                                 # print the summary of the model created

        epochs = 1
        print("[INFO] training network...")
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

        # train or fit the model to the data using k cross validation
        kf = KFold(n_splits=10, random_state=1)
        print("[INFO] Splitting data into ", kf, "...")
        history = []
        for a, b in kf.split(trainX, trainY):           # takes 10% of data for validation
            H = model.fit(trainX[a], trainY[a], batch_size = 32, validation_data=(trainX[b], trainY[b]), epochs=epochs)
            history.append(H)

        # evaluate the network
        print("[INFO] evaluating network...")
        predictions = model.predict(testX, batch_size = 5)
        print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

        # plot the training loss and accuracy
        # doesn't work properly
        N = np.arange(0, epochs)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H.history["loss"], label="train_loss")
        plt.plot(N, H.history["val_loss"], label="val_loss")
        plt.plot(N, H.history["accuracy"], label="train_acc")
        plt.plot(N, H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy (Simple NN)")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(self.dataPath+"plot.png")

        print("[INFO] evaluation done...")
        
        # save and reload the trained model and label binarizer
        model.save(self.dataPath+'newModel.h5')
        self.saveLB(self.dataPath+"lb", lb)
        print("[INFO] serializing network and label binarizer done...")

        self.model = tf.keras.models.load_model(self.dataPath+"newModel.h5")
        self.lb = self.load(self.dataPath+"lb")
        print("[INFO] reloading of model successful...")

    def addUser(self):
        """
            takes username, takes face pictures of the user present, and saves them
            counts 100
            if no face detected at all since start, then does nthg
            if a face was detected, add the same face if no new is detected
        """
        if not self.nameEntry.get():                                    # if not True i.e empty
            print("[INFO] Empty User Name!")
        else:
            if self.lastFace is not None:                               # for first use if no face at all
                if self.count == 0:
                    self.uname = self.nameEntry.get()
                    print("[INFO] adding User..."+self.uname)
                    self.dataPathImg = "./dataset/" + self.uname + "/"
                if not os.path.exists(self.dataPathImg):
                    os.makedirs(self.dataPathImg)
                cv2.imwrite(self.dataPathImg + self.uname + '.' + str(self.count) + ".jpg", cv2.resize(self.lastFace, (30, 30)))
                self.count+=1
            else:
                print("[INFO] no face detected...")

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()                                               # release web camera
        cv2.destroyAllWindows()                                         # it is not mandatory in this application

    # methods for saving and loading the binarizer
    def saveLB(self, filename, lb):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename + ".pkl", 'wb') as f:
            f.write(pickle.dumps(lb))
            f.close()

    def load(self, filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename + ".pkl", 'rb') as f:
            return pickle.load(f)
            
    """
        Functions for Servo movements
    """       
    def moveServoPos(self, i, p):
        '''
            moves the servo positively to the right or down
            args:
                i:  duty cycle
                p:  pwm
            return:
                i:  duty cycle
        '''
        p.ChangeDutyCycle(i)
        return (i+.1)
        
    def moveServoNeg(self, i, p):
        '''
            moves the servo negatively to the left or up
            args:
                i:  duty cycle
                p:  pwm
            return:
                i:  duty cycle
        '''
        p.ChangeDutyCycle(i)
        return (i-.1)

    def verPosition(self, iVer):
        '''
            Sets vertical servo positon 
        '''
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        servoPIN = 17
        GPIO.setup(servoPIN, GPIO.OUT)
        p = GPIO.PWM(servoPIN, 50)
        p.start(iVer)
        time.sleep(1)                                                   # add time delay to let servo process the code
        p.stop()

    def position(self, i, iVer):
        '''
            Initial position of the servo to start
            args:
                i, iVer:    position servo at i and iVer; horizontal and vertical duty cycle
        '''
        self.verPosition(iVer)

        servoPIN = 22
        GPIO.setup(servoPIN, GPIO.OUT)
        q = GPIO.PWM(servoPIN, 50)
        q.start(i)
        print("Initial Position Set...")
        time.sleep(1)
        q.stop()
        
        GPIO.cleanup()
        
    def mapServoFace(self, x, y, w, h):
        '''
            maps the servo position to face position
            executes only if the face goes out of 90% of the frame boundary
            args:
                x, y:   face top left cordinates
                w, h:   face width, height
                i:      horizontal duty cycle value
            return:
                i:      horizontal duty cycle value to keep track of horizontal movement
        '''
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        if (x < 0.1*self.fLen or x+w > 0.9*self.fLen):                  # horizontal movement
            GPIO.setup(22, GPIO.OUT)
            p = GPIO.PWM(22, 50)
            p.start(self.i)
            if (x < 0.1*self.fLen):                                     # move left, neg movement, r
                #print("x<")
                if self.i > 12.0:
                    self.i = 12.0
                else:
                    self.i = self.moveServoPos(self.i, p)
                    time.sleep(.1)
            else:                                                       # move right, pos movement, l
                #print("x+w>")
                if self.i < 2.0:
                    self.i = 2.0
                else:
                    self.i = self.moveServoNeg(self.i, p)
                    time.sleep(.1)
            p.stop()
        if (y < 0.1*self.fWid or y+h > 0.9*self.fWid):                  # vertical movement
            GPIO.setup(17, GPIO.OUT)
            p = GPIO.PWM(17, 50)
            p.start(self.iVer)
            if (y < 0.1*self.fWid):                                     # move down, pos movement, u
                #print("y<")
                if self.iVer < 1.5:
                    self.iVer = 1.5
                else:
                    self.iVer = self.moveServoNeg(self.iVer, p)
                    time.sleep(.1)
            else:                                                       # move up, neg movement, d
                #print("y+h>")
                if self.iVer > 4.0:
                    self.iVer = 4.0
                else:
                    self.iVer = self.moveServoPos(self.iVer, p)
                    time.sleep(.1)
            p.stop()
        #print("Face mapped in frame")
        GPIO.cleanup()
            

# start the app
print("[INFO] starting...")
pba = Application()
pba.root.mainloop()

