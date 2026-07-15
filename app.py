import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="Detector de Plagas", layout="wide")

st.title("🍃 Detector de Plagas en Hojas")
st.markdown("### Modelo YOLO11s - mAP50: 82.7%")

# ==========================================
# CONFIGURACIÓN DE TELEGRAM - TUS DATOS
# ==========================================
TELEGRAM_BOT_TOKEN = "8725129241:AAGBYwVLnmVfbBUa9RVjIdQD2AaOswKjinc"
TELEGRAM_CHAT_ID = "7700414080"

def enviar_alerta_telegram(clase, confianza, imagen_bytes=None):
    """Envía alerta a Telegram solo si es crítico o nada saludable"""
    if clase not in ['Crítico', 'Nada Saludable']:
        return False
    
    # Zona horaria de Ecuador (UTC-5)
    ecuador_tz = timezone(timedelta(hours=-5))
    ahora = datetime.now(ecuador_tz)
    
    mensaje = f"""
🚨 *ALERTA DE PLAGA DETECTADA* 

🍃 *Clase:* {clase}
📊 *Confianza:* {confianza:.2f}%
 *Hora:* {ahora.strftime('%H:%M:%S')}
📅 *Fecha:* {ahora.strftime('%d/%m/%Y')}

⚠️ *Acción recomendada:* Revisar planta inmediatamente
    """
    
    try:
        # Enviar mensaje de texto
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": mensaje,
            "parse_mode": "Markdown"
        }, timeout=10)
        
        # Enviar imagen si está disponible
        if imagen_bytes:
            url_foto = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {'photo': imagen_bytes}
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": f"📸 Evidencia: {clase} - {confianza:.2f}%"
            }
            requests.post(url_foto, files=files, data=data, timeout=10)
        
        return True
    except Exception as e:
        st.error(f"Error enviando a Telegram: {e}")
        return False

# ==========================================
# CARGAR MODELO
# ==========================================
@st.cache_resource
def load_model():
    model_url = "https://huggingface.co/EAMB2001/detector-plagas-modelo/resolve/main/modelo.pt"
    
    response = requests.get(model_url)
    model_path = "/tmp/modelo.pt"
    with open(model_path, "wb") as f:
        f.write(response.content)
    
    return YOLO(model_path)

try:
    model = load_model()
    CLASSES = ['Crítico', 'Nada Saludable', 'Saludable', 'media_saludable']
except Exception as e:
    st.error(f"Error cargando el modelo: {e}")
    model = None

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================
uploaded_file = st.file_uploader(" Sube una imagen de hoja", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Imagen original", use_column_width=True)
    
    if st.button("🔍 Analizar Hoja"):
        with st.spinner("Analizando imagen..."):
            image = Image.open(uploaded_file)
            temp_path = "/tmp/temp_image.jpg"
            image.save(temp_path)
            
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
                    st.image(result_img, caption="Resultado", use_column_width=True)
                    
                    st.write("### 📊 Todas las detecciones:")
                    for box in boxes:
                        cls = CLASSES[int(box.cls[0])]
                        confidence = float(box.conf[0]) * 100
                        st.write(f"• **{cls}**: {confidence:.2f}%")
                    
                    # ==========================================
                    # ENVIAR ALERTA A TELEGRAM
                    # ==========================================
                    if clase in ['Crítico', 'Nada Saludable']:
                        img_byte_arr = BytesIO()
                        image.save(img_byte_arr, format='JPEG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        if enviar_alerta_telegram(clase, conf, img_byte_arr):
                            st.warning("️ **Alerta enviada a Telegram**")
                        else:
                            st.error("❌ Error al enviar alerta")
                    else:
                        st.info("✅ Detección normal - Sin alerta")
                        
                else:
                    st.warning("No se detectó ninguna hoja")

st.markdown("---")
st.markdown("""
### ℹ️ Información:
- **Alertas automáticas:** Se envían solo cuando se detecta 'Crítico' o 'Nada Saludable'
- **Modelo:** YOLO11s entrenado con mAP50: 82.7%
- **Zona horaria:** Ecuador (UTC-5)
""")
