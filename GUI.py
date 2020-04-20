from PIL import Image, ImageTk
import tkinter as tk
import threading
import cv2
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import Capstone.textAlert as textAlert
import Capstone.NN as NN


faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

class Application:
    def __init__(self, dataPath = "./output/"):
        """
            Initialize application which uses OpenCV + Tkinter. It displays video frame and buttons.
        """
        self.vs = cv2.VideoCapture(0)
        self.dataPath = dataPath  # store output path
        self.dataPathImg = None     # path to store the images
        self.current_image = None  # current image from the camera
        self.lastFace = np.zeros(shape=[30, 30, 1], dtype=np.uint8)
        self.count = 0
        self.uname = None
        self.rec = False            # flag to use recognizer or not
        self.targets = self.getTargets()
        self.now = abs(datetime.now().minute-5)

        # objects of all necessary classes
        self.alert = textAlert.Alert()
        self.nn = NN.FaceNeural(self.dataPath)

        self.model = tf.keras.models.load_model(self.dataPath+"newModel.h5")
        self.lb = self.nn.load(self.dataPath+"lb")
        print("[INFO] models loaded...")

        self.root = tk.Tk()  # initialize root window
        self.root.title("Face Detective")  # set window title

        # self.destructor function gets fired when the window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        frame1 = tk.Frame(self.root)
        frame1.pack(side="top")
        self.panel = tk.Label(frame1)  # initialize video panel
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

    def videoLoop(self):
        """
            Get frame from the video stream, detect a face, and show it in tkinter GUI
        """
        ref, frame = self.vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # face detection
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # saving the last face every time we have a face
            self.lastFace = gray[y:y+h, x:x+w]

        if self.count <= 100 and self.count != 0:
            self.addUser()
        if self.count > 100:
            print("[INFO] Face Added to folder")
            self.count = 0
            self.nameEntry.delete(0, 'end')

        if ref:
            frame = cv2.flip(frame, 1)
            if self.rec == True:    # started to recognize
                image = cv2.resize(cv2.equalizeHist(self.lastFace), (30, 30))
                image = image.astype("float") / 255.0
                image = image.reshape(1, *image.shape, 1)
                preds = self.model.predict(image)
                i = preds.argmax(axis=1)[0]
                label = self.lb.classes_[i]
                # TODO: Create and read from txt file to target users
                if (abs(self.now - datetime.now().minute) >= 1):        # check time first
                    # pass
                    print("textSENT")
                    # if label in self.targets:                           # check your target
                    #     cv2.imwrite("./output/lastFace.jpg", self.lastFace)
                    #     self.now = self.alert.sendText(label)

                # draw the class label + probability on the output image
                if preds[0][i]*100 <= 70:
                    label += "not recognized"
                text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, True)

            # displaying in the GUI
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            self.panel.config(image=imgtk)  # show the image
        self.root.after(30, self.videoLoop)  # call the same function after 30 milliseconds

    def toggelRec(self):
        if self.rec == True:
            self.rec = False
        else:
            self.rec = True

    def targetUser(self):
        """
            write, and close the txt file containing targets
        """
        if not self.nameEntry.get():        # if not True i.e empty
            print("[INFO] Empty User Name!")
        else:
            f = open("./output/targets.txt", "w+")
            f.write(self.nameEntry.get()+"\r")
            f.close()
            print("[INFO] target set...")
            self.nameEntry.delete(0, 'end')
            self.targets = self.getTargets()

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
            self.targets = self.getTargets()

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
        """
            Delete folder
            Change label to nuser
        :return:
        """
        pass


    def trainNeural(self):
        """
            takes the pictures taken and use it to train the neural network architecture model
            saves the trained model
        """
        self.nn.train()


    def addUser(self):
        """
            takes username, takes face pictures of the user present, and saves them
            counts 100
            if no face detected at all since start, then does nthg
            if a face was detected, add the same face if no new is detected
        """
        if not self.nameEntry.get():        # if not True i.e empty
            print("[INFO] Empty User Name!")
        else:
            if self.lastFace is not None:   # for first use if no face at all
                if self.count == 0:
                    self.uname = self.nameEntry.get()
                    print("[INFO] adding User..."+self.uname)
                    self.dataPathImg = "./dataset/" + self.uname + "/"
                if not os.path.exists(self.dataPathImg):
                    os.makedirs(self.dataPathImg)
                cv2.imwrite(self.dataPathImg + self.uname + '.' + str(self.count) + ".jpg", cv2.resize(cv2.equalizeHist(self.lastFace), (30, 30)))
                self.count+=1
            else:
                print("[INFO] no face detected...")

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

# start the app
print("[INFO] starting...")
pba = Application()
pba.root.mainloop()