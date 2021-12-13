from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import os


#model = load_model('butterfly_e20_1213_2.h5')
model = load_model('butterfly_e20_1213_2.h5')
cap = cv2.VideoCapture(0)#내 웹캠에 연결

while cap.isOpened():
    
    ret, img = cap.read()
    if not ret:
        break

    h, w = img.shape[:2]


    result_img = img.copy()

    x1 = int(w/4)
    y1 = int(h/4)
    x2 = int(w/4*3)
    y2 = int(h/4*3)


    img_middle = img[y1:y2, x1:x2] #1031 화면이 빠르게 움직이거나 얼굴이 2개면 에러 개선해라
    
    try:
        img_middle_input = cv2.resize(img_middle, dsize=(224, 224))
    except:
        break
    img_middle_input = cv2.cvtColor(img_middle_input, cv2.COLOR_BGR2RGB)
    img_middle_input = preprocess_input(img_middle_input)
    img_middle_input = np.expand_dims(img_middle_input, axis=0)
    
    img_middle1, img_middle2, img_middle3, img_middle4 = model.predict(img_middle_input).squeeze()# 학습 할 때 노랑-배추-호랑-나비가아님 순으로 학습함
    #img_middle1, img_middle2, img_middle3 = model.predict(img_middle_input).squeeze()
    label = str(max(img_middle1, img_middle2, img_middle3, img_middle4))
    #label = str(max(img_middle1, img_middle2, img_middle3))
    print(img_middle1,img_middle2,img_middle3,img_middle4)
    """
    if max(img_middle1, img_middle2, img_middle3, img_middle4) == img_middle4:
        label ="img_middle4" +  str(max(img_middle1, img_middle2, img_middle3, img_middle4))
    if max(img_middle1, img_middle2, img_middle3, img_middle4) == img_middle3:
        label ="img_middle3" +  str(max(img_middle1, img_middle2, img_middle3, img_middle4))
    if max(img_middle1, img_middle2, img_middle3, img_middle4) == img_middle2:
        label ="img_middle2" +  str(max(img_middle1, img_middle2, img_middle3, img_middle4))
    if max(img_middle1, img_middle2, img_middle3, img_middle4) == img_middle1:
        label ="img_middle1" +  str(max(img_middle1, img_middle2, img_middle3, img_middle4))
        """
    
    
    fontpath = "fonts/gulim.ttc"
    font = ImageFont.truetype(fontpath, 30)
    img_pil = Image.fromarray(result_img)
    draw = ImageDraw.Draw(img_pil)
    """
    if max(img_middle1, img_middle2, img_middle3) == img_middle3:
        label ="호랑나비속  " +  str(max(img_middle1, img_middle2, img_middle3))
        color = (255, 200, 50)
        b,g,r,a = 255,200,50,10

    if max(img_middle1, img_middle2, img_middle3) == img_middle2:
        label ="배추흰나비속 " +  str(max(img_middle1, img_middle2, img_middle3))
        color = (230, 230, 230)
        b,g,r,a = 230,230,230,10

    if max(img_middle1, img_middle2, img_middle3) == img_middle1:
        label ="노랑나비속" +  str(max(img_middle1, img_middle2, img_middle3))
        color = (0, 103, 163)
        b,g,r,a = 0,103,163,10
    """    
    if max(img_middle1, img_middle2, img_middle3, img_middle4) == img_middle4:
        label ="호랑나비속 " +  str(max(img_middle1, img_middle2, img_middle3, img_middle4))
        color = (255, 200, 50)
        b,g,r,a = 255,200,50,10
    if max(img_middle1, img_middle2, img_middle3, img_middle4) == img_middle3:
        label ="배추흰나비속 " +  str(max(img_middle1, img_middle2, img_middle3, img_middle4))
        color = (230, 230, 230)
        b,g,r,a = 230,230,230,10
    if max(img_middle1, img_middle2, img_middle3, img_middle4) == img_middle2:
        label ="노랑나비속 " +  str(max(img_middle1, img_middle2, img_middle3, img_middle4))
        color = (0, 103, 163)
        b,g,r,a = 0,103,163,10 
    if max(img_middle1, img_middle2, img_middle3, img_middle4) == img_middle1:
        label ="나비가 아님 " +  str(max(img_middle1, img_middle2, img_middle3, img_middle4))
        color = (255, 30, 50)
        b,g,r,a = 255,30,50,10
    if max(img_middle1, img_middle2, img_middle3, img_middle4) < 0.95:
        label ="모르는 나비 "       # +  str(max(img_middle1, img_middle2, img_middle3, img_middle4))
        color = (30, 30, 200)
        b,g,r,a = 30,30,200,10
    #label = str(max(img_middle1, img_middle2, img_middle3))            
    #color = (100, 200, 100)

    
    #cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA) #텍스트 입히기
    
    #draw.text((x1 , y2+20), label, font=font, fill=(b,g,r,a),)
    #out.write(result_img)
    draw.text((x1, y2+20), label, font=font, fill=(b,g,r,a), stroke_width=2,stroke_fill="black")
    result_img = np.array(img_pil)
    cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA) #사각형 만들기
    
    cv2.imshow('result', result_img)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
