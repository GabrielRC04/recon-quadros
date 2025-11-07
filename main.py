from is_wire.core import Channel,Subscription,Message
from is_msgs.image_pb2 import Image
import numpy as np
import cv2

def to_np(input_image):
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image

# Carrega o dicionario que foi usado para gerar os ArUcos e inicializa o detector usando valores padroes para os parametros
parameters =  cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)

camera_id = 1

# Definicoes para aquisicao das imagens no Espaco Inteligente
broker_uri = "amqp://guest:guest@10.10.2.211:30000"
channel = Channel(broker_uri)

subscription = Subscription(channel=channel)
subscription.subscribe(topic='CameraGateway.{}.Frame'.format(camera_id))

nome_imagem = 'Camera'

while(True):
    # Captura um frame
    msg = channel.consume()
    if type(msg) != bool:
        img = msg.unpack(Image)
        frame = to_np(img)

    # Detecta os marcadores na imagem
    markerCorners, markerIds, rejectedImgPoints = arucoDetector.detectMarkers(frame)

    # Desenha as quinas detectadas na imagem
    img01_corners = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
    cv2.imshow('img01_corners',img01_corners)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
