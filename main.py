from is_wire.core import Channel,Subscription,Message
from is_msgs.image_pb2 import Image
import numpy as np
import cv2

# Função que consegue as imagens das camêras?
def to_np(input_image):
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image 

# Leitura das imagens e apresentacao
quadro01_rgb = cv2.imread('criacaodepindao.jpeg')
#quadro01_rgb = cv2.cvtColor(quadro01, cv2.COLOR_RGB2BGR)

# Tamanho da imagem que sera inserida no lugar dos ArUcos
[l,c,ch] = np.shape(quadro01_rgb)

# Pixels das quinas da imagem que sera inserida com ajuda do warp
pts_src = np.array([[0,0],[c,0],[c,l],[0,l]])

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

    # Detecta os marcadores na imagem (frame)
    markerCorners, markerIds, rejectedImgPoints = arucoDetector.detectMarkers(frame)
    
    # Para cada marcador detectado
    for mark in markerCorners:  
      # Anota as quinas do marcador detectado como pontos de destino da homografia
      pts_dst = np.array(mark[0])

      # Calcula a homografia
      H, status = cv2.findHomography(pts_src, pts_dst)

      # Faz o warp na imagem para que ela seja inserida
      warped_image = cv2.warpPerspective(quadro01_rgb, H, (frame.shape[1],frame.shape[0]))

      # Prepara a mascara para que apenas a foto contida no warp da imagem substitua pixels da outra imagem
      mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
      cv2.fillConvexPoly(mask, np.int32([pts_dst]), (1, 1, 1), cv2.LINE_AA)

      # Transforma essa mascara em 3 canais
      mask3 = np.zeros_like(warped_image)
      for i in range(0, 3):
          mask3[:,:,i] = mask

    # Desenha as quinas detectadas na imagem
    img01_corners = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
    
    # Insere a imagem do warp na imagem original tirando o ArUco dela com ajuda da mascara
    frame_masked = cv2.multiply(frame, 1-mask3)
    quadrop_rgb = cv2.add(warped_image, frame_masked)
    
    cv2.imshow('img01_corners',quadrop_rgb)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
