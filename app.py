import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime, timezone, timedelta
import tempfile
import os

st.set_page_config(page_title="Detector de Plagas", layout="wide")

st.title("🍃 Detector de Mosca Blanca en Hojas de Algodón")
st.markdown("### By: Erick Mera - Kevin Garcia")

# ==========================================
# CONFIGURACIÓN DE TELEGRAM (SOLO PARA ENVIAR ALERTAS)
# ==========================================
TELEGRAM_BOT_TOKEN = "8725129241:AAGBYwVLnmVfbBUa9RVjIdQD2AaOswKjinc"
TELEGRAM_CHAT_ID = "7700414080"

# Zona horaria Ecuador (UTC-5)
ecuador_tz = timezone(timedelta(hours=-5))

# ==========================================
# FUNCIÓN PARA ENVIAR ALERTAS A TELEGRAM
# ==========================================
def enviar_alerta_telegram(clase, conf, imagen_bytes):
    """Envía alerta SOLO si es crítico o nada saludable"""
    if clase not in ['Crítico', 'Nada Saludable']:
        return
    
    ahora = datetime.now(ecuador_tz)
    mensaje = f"""
 *ALERTA DE PLAGA DETECTADA*

🍃 *Clase:* {clase}
📊 *Confianza:* {conf:.2f}%
⏰ *Hora:* {ahora.strftime('%H:%M:%S')}
 *Fecha:* {ahora.strftime('%d/%m/%Y')}

⚠️ *Acción recomendada:* Revisar planta inmediatamente
    """
    
    try:
        # PRIMERO: Enviar mensaje de texto
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        response_text = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": mensaje,
            "parse_mode": "Markdown"
        }, timeout=10)
        
        print(f"✅ Mensaje enviado: {response_text.status_code}")
        
        # SEGUNDO: Enviar imagen DEBAJO del mensaje
        if imagen_bytes and len(imagen_bytes) > 0:
            url_foto = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {'photo': ('image.jpg', imagen_bytes, 'image/jpeg')}
            caption = f"📸 Evidencia: {clase} - {conf:.2f}%"
            
            response_photo = requests.post(url_foto, files=files, 
                                          data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, 
                                          timeout=10)
            
            print(f"📸 Imagen enviada: {response_photo.status_code}")
            if response_photo.status_code != 200:
                print(f"❌ Error enviando imagen: {response_photo.text}")
        else:
            print("️ No hay imagen para enviar")
            
    except Exception as e:
        print(f"❌ Error enviando a Telegram: {e}")

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
# INTERFAZ WEB
# ==========================================
st.sidebar.info("""
** Instrucciones:**
1. Sube una imagen de hoja de algodón
2. Haz clic en 'Analizar Hoja'
3. Si es 'Crítico' o 'Nada Saludable', recibirás alerta en Telegram con la imagen
""")

uploaded_file = st.file_uploader("📷 Sube una imagen de hoja", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Imagen original", width=500)
    
    if st.button("🔍 Analizar Hoja"):
        with st.spinner("Analizando imagen..."):
            try:
                # Abrir imagen
                image = Image.open(uploaded_file)
                
                # Convertir a RGB si es necesario (soluciona error de PIL)
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                elif image.mode == 'P':
                    image = image.convert('RGB')
                
                # Crear archivo temporal
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    temp_path = tmp_file.name
                    image.save(temp_path, format='JPEG')
                
                # Ejecutar detección
                results = model(temp_path, verbose=False)
                boxes = results[0].boxes
                
                with col2:
                    if len(boxes) > 0:
                        mejor = max(boxes, key=lambda b: float(b.conf[0]))
                        clase = CLASSES[int(mejor.cls[0])]
                        conf = float(mejor.conf[0]) * 100
                        
                        st.success(f"✅ **{clase}**")
                        st.metric("Confianza", f"{conf:.2f}%")
                        
                        # Imagen con detecciones
                        result_img = results[0].plot()
                        st.image(result_img, caption="Resultado", width=500)
                        
                        st.write("### 📊 Todas las detecciones:")
                        for box in boxes:
                            cls = CLASSES[int(box.cls[0])]
                            confidence = float(box.conf[0]) * 100
                            st.write(f"• **{cls}**: {confidence:.2f}%")
                        
                        # Enviar alerta si es crítico
                        if clase in ['Crítico', 'Nada Saludable']:
                            img_bytes = BytesIO()
                            image.save(img_bytes, format='JPEG', quality=85)
                            img_bytes.seek(0)
                            enviar_alerta_telegram(clase, conf, img_bytes.getvalue())
                            st.warning("⚠️ **Alerta enviada a Telegram**")
                    else:
                        st.warning("No se detectó ninguna hoja en la imagen")
                
                # Limpiar archivo temporal
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                st.error(f"❌ Error procesando la imagen: {e}")
                st.error("Asegúrate de que la imagen sea válida (JPG, PNG o JPEG)")

st.markdown("---")
st.markdown("""
### ℹ️ Información:
- **Alertas automáticas:** Se envían cuando se detecta 'Crítico' o 'Nada Saludable' en alguna hoja 
- **Zona horaria:** Ecuador (UTC-5)
""")
