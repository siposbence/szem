import cv2
import sys
import numpy as np
import face_recognition
import screeninfo
import random
import time 
import glob

class watch_me:
    
    
    def __init__(self, file_eye = 'szem_2.png', file_pupil = 'pupilla_2.png'):
        #self.eye = cv2.cvtColor(cv2.imread(file_eye, cv2.IMREAD_UNCHANGED), cv2.COLOR_RGBA2RGB)
        self.eye = cv2.imread(file_eye, cv2.IMREAD_UNCHANGED)
        self.pupil = cv2.imread(file_pupil, cv2.IMREAD_UNCHANGED)
        self.emotion = False
        self.current_emotion = "blink"
        self.type = 0
        self.emotion_set = {"s":"sad_1","d":"sad_2","l":"laugh","b":"blink", "a":"suprised"}
        self.n = 0
        self.n_max = 1
        self.starting_period = False
        self.start_x = 0
        self.start_y = 0
        self.smoothing = 15
        self.new_x = False
        
        
    def start_emotion(self, character):
        print("start"+character)
        if character in ['s','d','l','b','a']:
            self.emotion = True
            if self.n == 0:
                self.starting_period = True
                self.current_emotion = self.emotion_set[character]
                self.n_max = len(glob.glob(self.current_emotion+"/*"))

    def draw_eye(self, x, y):
        if (self.emotion and not self.starting_period):
            self.n+=1
            if self.n>self.n_max:
                self.n = 0
                self.emotion = False
                self.new_x = True
                print(self.new_x)
                image = cv2.imread(self.current_emotion+"/"+str(self.n_max)+".jpg")
            else:
                image = cv2.imread(self.current_emotion+"/"+str(self.n)+".jpg")
        else:
            if self.starting_period:
                if self.n == 0:
                    self.start_x = x
                    self.start_y = y
                self.n+=1
                x,y = self.start_x*(self.smoothing-self.n)/self.smoothing, -37*(self.n)/self.smoothing+self.start_y*(self.smoothing-self.n)/self.smoothing
                x,y = int(x), int(y) #offset
                if self.n >self.smoothing:
                    self.starting_period = False
            pupilla = np.roll(np.roll(self.pupil, x, axis=1), y, axis=0)
            #pupilla = np.roll(self.pupil, y, axis=0)

            ret,mask = cv2.threshold(pupilla[:,:,3],25,255,cv2.THRESH_BINARY)
            ret,mask1 = cv2.threshold(pupilla[:,:,3],25,255,cv2.THRESH_BINARY_INV )
            #pupilla = cv2.cvtColor(pupilla, cv2.COLOR_RGBA2RGB)
            B, G, R, A = cv2.split((np.array(cv2.bitwise_and(self.eye,self.eye,mask = mask1))+np.array(cv2.bitwise_and(pupilla,pupilla,mask = mask))))
            alpha = A / 255

            R = (255 * (1 - alpha) + R * alpha).astype(np.uint8)
            G = (255 * (1 - alpha) + G * alpha).astype(np.uint8)
            B = (255 * (1 - alpha) + B * alpha).astype(np.uint8)

            image = cv2.merge((B, G, R))
        
        if self.new_x:
            self.new_x = False
            return image, True
        else:
            return image, False



    
# pislantas kepek
feligcsukva = cv2.imread("feligcsukva.png",0)
felignyitva = cv2.imread("felignyitva.png",0)
csukva = cv2.imread("csukva.png",0)

pupil = cv2.imread("pupil4.png")

# meret szorzo
multi = 2

# pislogas
blink_frec = 150
