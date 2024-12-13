import cv2
from ultralytics import YOLO

# Caminhos do modelo e da imagem
model_path = r'C:\Users\kauem\Documents\TCC\best.pt'
image_path = r'C:\Users\kauem\Documents\TCC\teste.jpg'

# Carrega o modelo
model = YOLO(model_path)

# Faz a predição
results = model.predict(source=image_path, show=True, save=True, show_labels=True, show_conf=False, conf=0.4, iou=0.7, save_txt=True)

# Carrega a imagem para desenhar
image = cv2.imread(image_path)

# Inicializa os diâmetros
optic_nerve_diameter = None
excavation_diameter = None

# Itera sobre os resultados
for r in results:
    boxes = r.boxes.xyxy  # Coordenadas das bounding boxes [x_min, y_min, x_max, y_max]
    classes = r.boxes.cls  # Classes associadas às bounding boxes
    
    for box, cls in zip(boxes, classes):
        x_min, y_min, x_max, y_max = map(int, box)  # Converte para inteiros
        
        # Calcula o ponto central horizontal e os extremos (mais alto e mais baixo)
        x_center = (x_min + x_max) // 2
        top_point = (x_center, y_min)  # Ponto mais alto
        bottom_point = (x_center, y_max)  # Ponto mais baixo

        # Desenha a linha entre os extremos
        if cls == 0:  # Substitua '0' pelo ID da classe do nervo óptico
            optic_nerve_diameter = bottom_point[1] - top_point[1]  # Altura da bounding box
            cv2.line(image, top_point, bottom_point, (0, 255, 0), 2)  # Verde para nervo óptico
        elif cls == 1:  # Substitua '1' pelo ID da classe da escavação
            excavation_diameter = bottom_point[1] - top_point[1]  # Altura da bounding box
            cv2.line(image, top_point, bottom_point, (255, 0, 0), 2)  # Azul para escavação

# Verifica se os diâmetros foram encontrados e calcula a proporção
if optic_nerve_diameter and excavation_diameter:
    proportion = excavation_diameter / optic_nerve_diameter
    print(f"Diâmetro do Nervo Óptico: {optic_nerve_diameter}px")
    print(f"Diâmetro da Escavação: {excavation_diameter}px")
    print(f"Proporção (Escavação / Nervo Óptico): {proportion:.2f}")

    # Adiciona a proporção na imagem
    text = f"Proporcao ED: {proportion:.2f}"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # Texto em amarelo
else:
    print("Erro: Não foi possível calcular os diâmetros. Verifique as classes ou as detecções.")
    text = "Proporcao ED: N/A"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Texto em vermelho

# Exibe a imagem com as linhas e a proporção
cv2.imwrite(r'C:\Users\kauem\Documents\TCC\test.jpg', image)
