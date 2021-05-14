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
            pupilla = np.roll(self.pupil, x, axis=1)
            pupilla = np.roll(pupilla, y, axis=0)

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

# video beolvasas
video_capture = cv2.VideoCapture(0)

# szem pozicio
x_list = []
y_list = []
no_face = 0
blink = random.randint(20,30)
blink_seq = [felignyitva, feligcsukva, csukva, csukva, feligcsukva, felignyitva]

# megjeleníto kepernyo
screen_id = 0
screen = screeninfo.get_monitors()[screen_id]
width, height = screen.width, screen.height
print(width, height)

window_name = 'projector'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

w = watch_me()

while True:
    # alap kep
    img = np.zeros((width, height, 3))+255
    
    # kep olvasas
    ret, frame = video_capture.read()
    width_cap, height_cap, tmp = frame.shape
    
    small_frame = cv2.resize(frame, (0, 0), fx=1/multi, fy=1/multi)
    rgb_small_frame = small_frame[:, :, ::-1]
    faces = face_recognition.face_locations(rgb_small_frame)
    face_size = []
    
    for (top, right, bottom, left) in faces:
        top *= multi
        right *= multi
        bottom *= multi
        left *= multi
        face_size.append(bottom-top)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
    if len(face_size)>0:
        top, right, bottom, left = faces[np.argmax(face_size)]
        cv2.rectangle(frame, (left*multi, top*multi), (right*multi, bottom*multi), (255, 255, 255), 2)
        x_list.append(left+right)
        y_list.append(top+bottom)
        if len(x_list)>10:
            x_list.pop(0)
            y_list.pop(0)
    else:    
        no_face+=1
        if no_face > 50:
            x_list.append((width_cap+11)//2)
            y_list.append((height_cap-125)//2)
            x_list.pop(0)
            y_list.pop(0)
            #print(x_list)
            
    try:
        image_L,_ = w.draw_eye(int(50-100*(np.mean(x_list))/width_cap)+20, int(50+100*(np.mean(y_list))/height_cap)-100)
        image_R,_ = w.draw_eye(-int(50-100*(np.mean(x_list))/width_cap)+20, int(50+100*(np.mean(y_list))/height_cap)-100)
        image = np.hstack([ np.flip(image_R,1), image_L])
        #np.roll(np.roll(pupil, int(140-280*(np.mean(x_list))/width_cap)+50, axis = 1), int(-140+280*(np.mean(y_list))/height_cap)+50, axis = 0)
        #cv2.circle(img,(600*(int(640-np.mean(x_list)))//width_cap-50, 600*(int(np.mean(y_list)))//height_cap+250), 100, (0,0,0), -1)
    except ValueError:
        image, _ = w.draw_eye(20,0)
        print(image.shape)
        image = np.hstack([np.flip(image,1), image])
        #shift(pupil, (0,0,0), cval=255) #cv2.circle(img,(411,411), 100, (0,0,0), -1)
        print("Nincs arc....!")
        
    # debughoz
    cv2.imshow('Video', frame)
    
    print(blink)
    blink -=1
    if blink < 1:
        image = cv2.bitwise_and(image,image,mask = blink_seq[-blink])
        time.sleep(0.05)
        if blink < -4:
            print("blink")
            blink = random.randint(blink_frec,blink_frec+50)
        
    cv2.imshow(window_name, image)
    key = cv2.waitKey(1)
    if key>0:
        w.start_emotion(chr(key))
    if key & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
