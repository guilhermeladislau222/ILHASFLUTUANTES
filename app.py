import streamlit as st
from PIL import Image
import numpy as np

def analyze_image(image):
    # Converter a imagem para escala de cinza
    gray_image = image.convert('L')
    
    # Converter para array numpy
    img_array = np.array(gray_image)
    
    # Calcular a média dos valores de pixel
    mean_value = np.mean(img_array)
    
    # Lógica simplificada: se a média for baixa, consideramos como possível presença de lixo
    # (isto é uma simplificação e não é precisa para detecção real de lixo)
    if mean_value < 100:  # Este valor limite pode ser ajustado
        return "Possível presença de lixo detectada!", mean_value
    else:
        return "Nenhum indício significativo de lixo detectado.", mean_value

st.title("Monitoramento Simplificado de Eco Barreira")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem Original", use_column_width=True)
    
    if st.button("Analisar Imagem"):
        result, value = analyze_image(image)
        st.write(result)
        st.write(f"Valor médio de pixel: {value:.2f}")
        
        if "Possível presença de lixo" in result:
            st.warning(result)
        else:
            st.success(result)

st.write("Nota: Esta é uma análise muito simplificada e não é precisa para detecção real de lixo. Serve apenas como demonstração.")
