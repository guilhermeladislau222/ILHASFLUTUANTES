import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

# Carregar o modelo de detecção de objetos
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
    return model

def detect_objects(image, model):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()
    
    return boxes, classes, scores

def draw_boxes(image, boxes, classes, scores, threshold=0.5):
    height, width, _ = image.shape
    for i in range(len(scores)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            ymin = int(ymin * height)
            xmin = int(xmin * width)
            ymax = int(ymax * height)
            xmax = int(xmax * width)
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, f"Class {classes[i]}: {scores[i]:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

st.title("Monitoramento de Eco Barreira com IA")

model = load_model()

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    st.image(image, caption="Imagem Original", use_column_width=True)
    
    if st.button("Analisar Imagem"):
        boxes, classes, scores = detect_objects(image_np, model)
        
        image_with_boxes = draw_boxes(image_np.copy(), boxes, classes, scores)
        st.image(image_with_boxes, caption="Imagem com Detecções", use_column_width=True)
        
        # Contar objetos relevantes (exemplo: classes 1-5 como lixo)
        relevant_objects = sum(1 for c, s in zip(classes, scores) if c <= 5 and s > 0.5)
        st.write(f"Objetos relevantes detectados: {relevant_objects}")
        
        if relevant_objects > 0:
            st.warning("Possível presença de lixo detectada na eco barreira!")
        else:
            st.success("Nenhum objeto relevante detectado. A eco barreira parece limpa!")

st.write("Nota: Este é um protótipo simples. A precisão pode variar dependendo das condições da imagem.")
