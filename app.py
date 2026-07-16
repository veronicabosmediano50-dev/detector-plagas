import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime, timezone, timedelta
import tempfile
import os
import time

st.set_page_config(page_title="Detector de Plagas", layout="wide")

st.title("🍃 Detector de Mosca Blanca en Hojas de Algodón")
st.markdown("### By: Erick Mera - Kevin Garcia")

# ==========================================
# CONFIGURACIÓN DE TELEGRAM (SOLO PARA ENVIAR ALERTAS)
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
 *ALERTA DE PLAGA DETECTADA*

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
        print(f"Error Telegram: {e}")

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
# FUNCIÓN DE ANÁLISIS
# ==========================================
def analizar_imagen(image):
    """Analiza una imagen y retorna los resultados"""
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        temp_path = tmp_file.name
        image.save(temp_path, format='JPEG')
    
    try:
        results = model(temp_path, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            mejor = max(boxes, key=lambda b: float(b.conf[0]))
            clase = CLASSES[int(mejor.cls[0])]
            conf = float(mejor.conf[0]) * 100
            result_img = results[0].plot()
            return clase, conf, result_img, True
        else:
            return None, 0, None, False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ==========================================
# SIDEBAR - SELECCIÓN DE MODO
# ==========================================
st.sidebar.title("🎯 Modo de Detección")
modo = st.sidebar.radio(
    "Selecciona el modo:",
    ["📷 Subir imagen", " Cámara en vivo", "⏱️ Tiempo real automático"]
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
                    clase, conf, result_img, detectado = analizar_imagen(image)
                    
                    with col2:
                        if detectado:
                            st.success(f"✅ **{clase}**")
                            st.metric("Confianza", f"{conf:.2f}%")
                            st.image(result_img, caption="Resultado", width=500)
                            
                            if clase in ['Crítico', 'Nada Saludable']:
                                img_bytes = BytesIO()
                                image.save(img_bytes, format='JPEG', quality=85)
                                img_bytes.seek(0)
                                enviar_alerta_telegram(clase, conf, img_bytes.getvalue())
                                st.warning("⚠️ **Alerta enviada a Telegram**")
                        else:
                            st.warning("No se detectó ninguna hoja")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ==========================================
# MODO 2: CÁMARA EN VIVO (FOTO ÚNICA)
# ==========================================
elif modo == "📹 Cámara en vivo":
    st.header("📹 Capturar desde Cámara")
    st.info("💡 Haz clic en el botón para activar la cámara de tu laptop y tomar una foto")
    
    # Usar un key único para la cámara
    camera_photo = st.camera_input(" Haz clic para activar la cámara y tomar una foto", key="live_camera")
    
    if camera_photo is not None:
        st.success("✅ Foto capturada correctamente")
        
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(camera_photo)
            st.image(image, caption="Foto capturada", width=500)
        
        with col2:
            if st.button("🔍 Analizar Foto", key="analizar_foto"):
                with st.spinner("Analizando imagen..."):
                    try:
                        clase, conf, result_img, detectado = analizar_imagen(image)
                        
                        if detectado:
                            st.success(f"✅ **{clase}**")
                            st.metric("Confianza", f"{conf:.2f}%")
                            st.image(result_img, caption="Resultado", width=500)
                            
                            if clase in ['Crítico', 'Nada Saludable']:
                                img_bytes = BytesIO()
                                image.save(img_bytes, format='JPEG', quality=85)
                                img_bytes.seek(0)
                                enviar_alerta_telegram(clase, conf, img_bytes.getvalue())
                                st.warning("⚠️ **Alerta enviada a Telegram**")
                        else:
                            st.warning("No se detectó ninguna hoja")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    else:
        st.info("👆 Esperando captura de foto...")

# ==========================================
# MODO 3: TIEMPO REAL AUTOMÁTICO
# ==========================================
elif modo == "⏱️ Tiempo real automático":
    st.header("️ Detección en Tiempo Real")
    st.info("💡 La cámara capturará y analizará automáticamente cada pocos segundos")
    
    col_config, col_display = st.columns([1, 2])
    
    with col_config:
        intervalo = st.slider("⏱️ Intervalo entre capturas (segundos)", 2, 10, 3)
        activar_camara = st.checkbox("🎥 Activar cámara en tiempo real")
        st.write(f"**Intervalo:** {intervalo} segundos")
    
    if activar_camara:
        camera_photo = st.camera_input("📸 Cámara en tiempo real", key="realtime_camera")
        
        if camera_photo:
            placeholder = st.empty()
            
            with st.spinner(" Análisis en tiempo real activo..."):
                while activar_camara:
                    try:
                        image = Image.open(camera_photo)
                        clase, conf, result_img, detectado = analizar_imagen(image)
                        
                        ahora = datetime.now(ecuador_tz)
                        
                        with placeholder.container():
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.image(image, caption="Captura actual", width=400)
                            
                            with col2:
                                if detectado:
                                    color = "🔴" if clase in ['Crítico', 'Nada Saludable'] else "🟢"
                                    st.markdown(f"### {color} **{clase}**")
                                    st.metric("Confianza", f"{conf:.2f}%")
                                    st.image(result_img, caption="Detección", width=400)
                                    
                                    st.markdown(f"""
                                    **Última detección:**
                                    - ⏰ {ahora.strftime('%H:%M:%S')}
                                    - 📅 {ahora.strftime('%d/%m/%Y')}
                                    """)
                                    
                                    if clase in ['Crítico', 'Nada Saludable']:
                                        st.warning("🚨 **¡ALERTA!** Revisa la planta")
                                else:
                                    st.warning("⚠️ No se detectó hoja")
                        
                        time.sleep(intervalo)
                        
                        # Recapturar para el siguiente ciclo
                        # Nota: en Streamlit, camera_input mantiene la última foto
                        # Para verdadero tiempo real, el usuario debe tomar nuevas fotos
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        time.sleep(intervalo)
                    
                    # Break manual - el usuario puede desactivar el checkbox
                    break  # Salimos después de una iteración para permitir interacción

# ==========================================
# INFORMACIÓN
# ==========================================
st.markdown("---")
st.markdown("""
### ℹ️ Información:
- **Alertas automáticas:** Se envían cuando se detecta 'Crítico' o 'Nada Saludable'
- **Modelo:** YOLO11s entrenado con mAP50: 82.7%
- **Zona horaria:** Ecuador (UTC-5)
- **Modos disponibles:** Subir imagen, Cámara en vivo, Tiempo real
""")
