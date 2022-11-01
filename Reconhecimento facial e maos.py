import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0) # cria a conexao com a webcam

reconhecimento_maos = mp.solutions.hands
reconhecimento_rosto = mp.solutions.face_detection 
desenho = mp.solutions.drawing_utils
maos = reconhecimento_maos.Hands()
reconhecedor_rosto = reconhecimento_rosto.FaceDetection() 

if webcam.isOpened():
   
    validacao, frame = webcam.read()
  
    # loop infinito
    while validacao:
       
        validacao, frame = webcam.read()
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converte BGR em RGB 
        
        lista_maos = maos.process(frameRGB)
        if lista_maos.multi_hand_landmarks:
            for mao in lista_maos.multi_hand_landmarks:
                print(mao.landmark)
                desenho.draw_landmarks(frame, mao, reconhecimento_maos.HAND_CONNECTIONS) # desenha a mao

        lista_rostos = reconhecedor_rosto.process(frame)      
        if lista_rostos.detections:
            for rosto in lista_rostos.detections:
                desenho.draw_detection(frame, rosto) # desenha o rosto na imagem
        
        cv2.imshow("Video da Webcam", frame)

        tecla = cv2.waitKey(2)
        if tecla == 27:
            break

webcam.release()
cv2.destroyAllWindows()