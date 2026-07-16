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

st.title(" Detector de Mosca Blanca en Hojas de Algodón")
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
 *Hora:* {ahora.strftime('%H:%M:%S')}
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
        self.frame_count = 0
        
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Procesar cada 3 frames para mejor rendimiento
            self.frame_count += 1
            if self.frame_count % 3 != 0:
                return img
            
            # Convertir BGR a RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Ejecutar detección
            results = self.model(img_rgb, verbose=False, conf=0.25, iou=0.45)
            
            # Dibujar detecciones
            annotated_frame = results[0].plot()
            
            # Obtener mejor detección
            if len(results[0].boxes) > 0:
                mejor = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                clase = self.classes[int(mejor.cls[0])]
                conf = float(mejor.conf[0]) * 100
                
                # Mostrar info en la imagen
                cv2.putText(annotated_frame, f"{clase}: {conf:.1f}%", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.putText(annotated_frame, f"{clase}: {conf:.1f}%", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            
            return annotated_frame
        except Exception as e:
            print(f"Error en video processor: {e}")
            return frame.to_ndarray(format="bgr24")

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
# MODO 3: TIEMPO REAL (DETECCIÓN AUTOMÁTICA)
# ==========================================
elif modo == "🎥 Tiempo Real":
    st.header("🎥 Detección en Tiempo Real")
    st.info(" **Muestra una hoja a la cámara** y el modelo detectará automáticamente su estado de salud en tiempo real.")
    
    st.markdown("""
    ### Características:
    - ✅ **Detección automática** sin necesidad de tomar fotos
    - ✅ **Análisis continuo** frame por frame
    - ✅ **Visualización en vivo** con bounding boxes
    - ✅ **Resultados inmediatos** en la pantalla
    """)
    
    st.markdown("---")
    
    # Configuración robusta de ICE servers
    rtc_configuration = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
        ],
        "iceCandidatePoolSize": 10,
    }
    
    try:
        webrtc_ctx = webrtc_streamer(
            key="detector-plagas-realtime-v2",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15}
                },
                "audio": False
            },
            rtc_configuration=rtc_configuration,
            async_processing=True,
        )
        
        st.markdown("""
        **Instrucciones:**
        1. Permite el acceso a la cámara cuando el navegador lo solicite
        2. Haz clic en **"START"** para iniciar
        3. Muestra una hoja de algodón a la cámara
        4. Verás la detección automática con bounding box y clase
        5. El nombre y confianza aparecen en la esquina superior izquierda
        """)
        
        if webrtc_ctx.state == "RUNNING":
            st.success("📹 Cámara activa - Detección en tiempo real funcionando")
        elif webrtc_ctx.state == "STOPPED":
            st.info(" Haz clic en 'START' para activar la cámara")
        else:
            st.warning(f"⏳ Estado: {webrtc_ctx.state}")
            
    except Exception as e:
        st.error(f"❌ Error al iniciar: {e}")
        st.info(" **Nota:** Si no funciona, intenta usar Chrome o Edge, o ejecuta localmente")

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
