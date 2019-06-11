import sys
import dlib
import cv2
import openface
import face_recognition
import numpy as np

predictor_model = "shape_predictor_68_face_landmarks.dat"

erison_image = face_recognition.load_image_file("imagens/erison.jpg")
erison_face_encoding = face_recognition.face_encodings(erison_image)[0]

faces_conhecidas = [
    "Erison"
]

faces_encodings = [
    erison_face_encoding
]

# Atribui o modelo de HOG do Dlib
face_detector = dlib.get_frontal_face_detector()
#Atribui o modelo de de Landmarks
face_pose_predictor = dlib.shape_predictor(predictor_model)
#Atribui o modelo de faceAligner do Openface
face_aligner = openface.AlignDlib(predictor_model)

# Pega o nome da imagem a partir do console
file_name = sys.argv[1]
# Carrega a imagem
image = cv2.imread(file_name)

# Utiliza o HOG para encontrar o rostos
detected_faces = face_detector(image, 1)

for i, face_rect in enumerate(detected_faces):
    name = "Desconhecido"
    pose_landmarks = face_pose_predictor(image ,face_rect)

    alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    face_encodings = face_recognition.face_encodings(alignedFace)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(faces_encodings, face_encoding)

        face_distances = face_recognition.face_distance(faces_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = faces_conhecidas[best_match_index]


    left = face_rect.left()
    top = face_rect.top()
    right = face_rect.right()
    bottom = face_rect.bottom()

    # Desenha uma caixa em volta do rosto
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    # Inclui o nome da pessoa na caixa
    cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Mostra os resultados
cv2.imshow('Video', image)
cv2.waitKey(0)
cv2.destroyAllWindows()