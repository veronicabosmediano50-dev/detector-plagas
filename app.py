import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime, timezone, timedelta
import tempfile
import os
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

st.set_page_config(page_title="Detector de Plagas", layout="wide")

st.title("🍃 Detector de Mosca Blanca en Hojas de Algodón")
st.markdown("### By: Erick Mera - Kevin Garcia")

# ==========================================
# CONFIGURACIÓN DE TELEGRAM
# ==========================================
TELEGRAM_BOT_TOKEN = "8725129241:AAGBYwVLnmVfbBUa9RVjIdQD2AaOswKjinc"
TELEGRAM_CHAT_ID = "7700414080"
ecuador_tz = timezone(timedelta(hours=-5))

# ==========================================
# FUNCIÓN PARA ENVIAR ALERTAS A TELEGRAM
# ==========================================
def enviar_alerta_telegram(clase, conf, imagen_bytes):
    if clase not in ['Crítico', 'Nada Saludable']:
        return
    
    ahora = datetime.now(ecuador_tz)
    mensaje = f"""
🚨 *ALERTA DE PLAGA DETECTADA*

🍃 *Clase:* {clase}
📊 *Confianza:* {conf:.2f}%
⏰ *Hora:* {ahora.strftime('%H:%M:%S')}
📅 *Fecha:* {ahora.strftime('%d/%m/%Y')}

⚠️ *Acción recomendada:* Revisar planta inmediatamente
    """
    
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}, timeout=10)
        if imagen_bytes and len(imagen_bytes) > 0:
            files = {'photo': ('image.jpg', imagen_bytes, 'image/jpeg')}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto", 
                          files=files, data={"chat_id": TELEGRAM_CHAT_ID, "caption": f"📸 Evidencia: {clase} - {conf:.2f}%"}, timeout=10)
    except Exception as e:
        print(f" Error Telegram: {e}")

# ==========================================
# CARGAR MODELO
# ==========================================
@st.cache_resource
def load_model():
    try:
        model_url = "https://huggingface.co/EAMB2001/detector-plagas-modelo/resolve/main/modelo.pt"
        response = requests.get(model_url, timeout=60)
        model_path = "/tmp/modelo.pt"
        with open(model_path, "wb") as f:
            f.write(response.content)
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None

model = load_model()
CLASSES = ['Crítico', 'Nada Saludable', 'Saludable', 'media_saludable']

if model is None:
    st.error("❌ Error cargando el modelo")
    st.stop()

# ==========================================
# PROCESADOR DE VIDEO EN TIEMPO REAL
# ==========================================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.classes = CLASSES
        self.last_detection = None
        self.last_detection_time = None
        self.detection_count = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convertir BGR a RGB para YOLO
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Ejecutar detección
        results = self.model(img_rgb, verbose=False, conf=0.25)
        
        # Dibujar detecciones
        annotated_frame = results[0].plot()
        
        # Obtener mejor detección
        if len(results[0].boxes) > 0:
            mejor = max(results[0].boxes, key=lambda b: float(b.conf[0]))
            clase = self.classes[int(mejor.cls[0])]
            conf = float(mejor.conf[0]) * 100
            
            self.last_detection = (clase, conf)
            self.last_detection_time = datetime.now()
            self.detection_count += 1
            
            # Mostrar info en la imagen
            cv2.putText(annotated_frame, f"{clase}: {conf:.1f}%", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame

# ==========================================
# SIDEBAR - SELECCIÓN DE MODO
# ==========================================
st.sidebar.title(" Modo de Detección")
modo = st.sidebar.radio(
    "Selecciona el modo:",
    ["📷 Subir imagen", " Cámara en vivo", "🎥 Tiempo Real"]
)

st.sidebar.info("""
**Instrucciones:**
1. Elige un modo de detección
2. El modelo analizará la hoja
3. Si es 'Crítico' o 'Nada Saludable', recibirás alerta en Telegram
""")

# ==========================================
# MODO 1: SUBIR IMAGEN
# ==========================================
if modo == "📷 Subir imagen":
    st.header("📷 Subir Imagen desde Archivo")
    uploaded_file = st.file_uploader("Sube una imagen de hoja", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Imagen original", width=500)
        
        if st.button("🔍 Analizar Hoja"):
            with st.spinner("Analizando imagen..."):
                try:
                    image = Image.open(uploaded_file)
                    
                    if image.mode in ('RGBA', 'P', 'LA'):
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        if image.mode == 'P':
                            image = image.convert('RGBA')
                        if image.mode in ('RGBA', 'LA'):
                            background.paste(image, mask=image.split()[-1])
                        image = background
                    elif image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        temp_path = tmp_file.name
                        image.save(temp_path, format='JPEG', quality=95)
                    
                    results = model(temp_path, verbose=False)
                    boxes = results[0].boxes
                    
                    with col2:
                        if len(boxes) > 0:
                            mejor = max(boxes, key=lambda b: float(b.conf[0]))
                            clase = CLASSES[int(mejor.cls[0])]
                            conf = float(mejor.conf[0]) * 100
                            
                            st.success(f"✅ **{clase}**")
                            st.metric("Confianza", f"{conf:.2f}%")
                            
                            result_img = results[0].plot()
                            st.image(result_img, caption="Resultado", width=500)
                            
                            if clase in ['Crítico', 'Nada Saludable']:
                                img_bytes = BytesIO()
                                image.save(img_bytes, format='JPEG', quality=85)
                                img_bytes.seek(0)
                                enviar_alerta_telegram(clase, conf, img_bytes.getvalue())
                                st.warning("⚠️ **Alerta enviada a Telegram**")
                        else:
                            st.warning("No se detectó ninguna hoja")
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ==========================================
# MODO 2: CÁMARA EN VIVO (FOTO ÚNICA)
# ==========================================
elif modo == " Cámara en vivo":
    st.header(" Capturar desde Cámara")
    st.info("💡 Haz clic en el botón de abajo, permite el acceso a la cámara y toma una foto.")
    
    img_file_buffer = st.camera_input("📸 Haz clic aquí para activar la cámara y tomar una foto")
    
    if img_file_buffer is not None:
        st.success("✅ Foto capturada. Analizando...")
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(img_file_buffer)
            st.image(image, caption="Foto capturada", width=500)
        
        with col2:
            try:
                if image.mode in ('RGBA', 'P', 'LA'):
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'P':
                        image = image.convert('RGBA')
                    if image.mode in ('RGBA', 'LA'):
                        background.paste(image, mask=image.split()[-1])
                    image = background
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    temp_path = tmp_file.name
                    image.save(temp_path, format='JPEG', quality=95)
                
                results = model(temp_path, verbose=False)
                boxes = results[0].boxes
                
                if len(boxes) > 0:
                    mejor = max(boxes, key=lambda b: float(b.conf[0]))
                    clase = CLASSES[int(mejor.cls[0])]
                    conf = float(mejor.conf[0]) * 100
                    
                    st.success(f"✅ **{clase}**")
                    st.metric("Confianza", f"{conf:.2f}%")
                    
                    result_img = results[0].plot()
                    st.image(result_img, caption="Resultado", width=500)
                    
                    if clase in ['Crítico', 'Nada Saludable']:
                        img_bytes = BytesIO()
                        image.save(img_bytes, format='JPEG', quality=85)
                        img_bytes.seek(0)
                        enviar_alerta_telegram(clase, conf, img_bytes.getvalue())
                        st.warning("⚠️ **Alerta enviada a Telegram**")
                else:
                    st.warning("No se detectó ninguna hoja")
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ==========================================
# MODO 3: TIEMPO REAL (SIMULADO CON REFRESCO)
# ==========================================
elif modo == "🎥 Tiempo Real":
    st.header("🎥 Detección en Tiempo Real (Modo Simplificado)")
    st.info(" **Toma fotos consecutivas rápidamente** mostrando una hoja a la cámara")
    
    st.markdown("""
    ### Instrucciones:
    1. Muestra la hoja a la cámara
    2. Toma una foto
    3. El modelo la analiza automáticamente
    4. Repite el proceso para monitoreo continuo
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        img_file_buffer = st.camera_input("📸 Tomar foto", key="realtime_cam")
        
        if img_file_buffer:
            image = Image.open(img_file_buffer)
            st.image(image, caption="Foto capturada", width=400)
            
            # Análisis automático
            with st.spinner(" Analizando..."):
                try:
                    if image.mode in ('RGBA', 'P'):
                        image = image.convert('RGB')
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        temp_path = tmp.name
                        image.save(temp_path, format='JPEG')
                    
                    results = model(temp_path, verbose=False)
                    
                    if len(results[0].boxes) > 0:
                        mejor = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                        clase = CLASSES[int(mejor.cls[0])]
                        conf = float(mejor.conf[0]) * 100
                        
                        with col2:
                            st.success(f"✅ **{clase}**")
                            st.metric("Confianza", f"{conf:.2f}%")
                            
                            result_img = results[0].plot()
                            st.image(result_img, caption="Resultado", width=400)
                            
                            if clase in ['Crítico', 'Nada Saludable']:
                                st.error("🚨 **¡ALERTA!**")
                                img_bytes = BytesIO()
                                image.save(img_bytes, format='JPEG', quality=85)
                                img_bytes.seek(0)
                                enviar_alerta_telegram(clase, conf, img_bytes.getvalue())
                                st.warning("⚠️ Alerta enviada a Telegram")
                    else:
                        with col2:
                            st.warning(" No se detectó hoja")
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    st.error(f" Error: {e}")
# ==========================================
# INFORMACIÓN
# ==========================================
st.markdown("---")
st.markdown("""
### ℹ️ Información:
- **Alertas automáticas:** Se envían cuando se detecta 'Crítico' o 'Nada Saludable'
- **Modelo:** YOLO11s entrenado con mAP50: 82.7%
- **Zona horaria:** Ecuador (UTC-5)
- **Modos disponibles:** Subir imagen, Cámara en vivo, Tiempo Real
""")
