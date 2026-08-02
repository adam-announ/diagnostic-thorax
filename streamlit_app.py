import os, urllib.request, json
import streamlit as st
import numpy as np, tensorflow as tf, cv2
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image

# ==========================================================================
#  MODELE v3 (320x320 + CLAHE) sur Hugging Face
# ==========================================================================
MODEL_URL = "https://huggingface.co/akert123/densenet-thorax/resolve/main/densenet121_14maladies_v3.keras"
MODEL_PATH = "densenet121_14maladies_v3.keras"
THRESH_URL = "https://huggingface.co/akert123/densenet-thorax/resolve/main/thresholds.json"
THRESH_PATH = "thresholds.json"

IMG_SIZE = 320   # v3 entraine en 320

st.set_page_config(page_title="Diagnostic Thorax IA - Hopital Cheikh Zaid",
                   page_icon="🫁", layout="wide")

PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
               'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
               'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

FR = {'Atelectasis':'Atelectasie','Cardiomegaly':'Cardiomegalie',
      'Effusion':'Epanchement pleural','Infiltration':'Infiltrat','Mass':'Masse',
      'Nodule':'Nodule','Pneumonia':'Pneumonie','Pneumothorax':'Pneumothorax',
      'Consolidation':'Consolidation','Edema':'Oedeme','Emphysema':'Emphyseme',
      'Fibrosis':'Fibrose','Pleural_Thickening':'Epaississement pleural','Hernia':'Hernie'}

# --------------------------------------------------------------- STYLE
st.markdown("""
<style>
    .stApp {background-color: #f4f7fb;}
    .bandeau {
        background: linear-gradient(90deg, #0b3d66 0%, #1a6fb0 100%);
        padding: 1.2rem 1.6rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;
    }
    .bandeau h1 {margin:0; font-size:1.5rem;}
    .bandeau p {margin:0; opacity:0.9; font-size:0.9rem;}
    .carte {background:white; border-radius:10px; padding:0.9rem 1.2rem;
            box-shadow:0 1px 5px rgba(0,0,0,0.08); margin-bottom:0.7rem;}
    .pos {color:#c0392b; font-weight:700;}
    .avert {background:#fff4e5; border-left:4px solid #ff9800; padding:0.8rem 1rem;
            border-radius:6px; font-size:0.85rem; color:#7a4b00; margin-top:1.2rem;}
    .stProgress > div > div > div {background-color:#1a6fb0;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------- EN-TETE
c1, c2 = st.columns([1, 6])
with c1:
    if os.path.exists("logo_hopital.png"):
        st.image("logo_hopital.png", width=110)
    else:
        st.markdown("<div style='width:110px;height:110px;border:2px dashed #b0bec5;"
                    "border-radius:12px;display:flex;align-items:center;justify-content:center;"
                    "color:#90a4ae;font-size:0.72rem;text-align:center;'>Logo<br>Hopital<br>"
                    "Cheikh Zaid</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='bandeau'><h1>🫁 Aide au diagnostic - Radiographies thoraciques</h1>"
                "<p>Detection automatique de 14 pathologies · DenseNet121 · Hopital Cheikh Zaid</p></div>",
                unsafe_allow_html=True)

# --------------------------------------------------------------- MODELE
@st.cache_resource
def charger_modele():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Telechargement du modele (une seule fois)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    dense_base = m.layers[0]
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    feat = dense_base(inp)
    x = feat
    for layer in m.layers[1:]:
        x = layer(x)
    grad = tf.keras.Model(inp, [feat, x])
    return m, grad

@st.cache_resource
def charger_seuils():
    try:
        if not os.path.exists(THRESH_PATH):
            urllib.request.urlretrieve(THRESH_URL, THRESH_PATH)
        with open(THRESH_PATH) as f:
            return json.load(f)
    except Exception:
        return {p: 0.5 for p in PATHOLOGIES}

model, grad_model = charger_modele()
SEUILS = charger_seuils()

# --------------------------------------------------------------- PRETRAITEMENT (CLAHE)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preparer(pil_img):
    g = np.array(pil_img.convert("L"))
    g = _clahe.apply(g.astype("uint8"))
    g = cv2.resize(g, (IMG_SIZE, IMG_SIZE))
    rgb = np.stack([g]*3, axis=-1)
    x = preprocess_input(rgb.astype("float32"))
    return np.expand_dims(x, 0), rgb

def est_radio(pil_img):
    arr = np.array(pil_img.convert("RGB")).astype("float32")
    r, gg, b = arr[...,0], arr[...,1], arr[...,2]
    ecart = (np.abs(r-gg).mean() + np.abs(gg-b).mean())/2
    h, w = arr.shape[:2]
    ratio = max(h,w)/min(h,w)
    contraste = arr.mean(-1).std()
    return ecart < 15 and ratio < 1.6 and contraste > 25

def gradcam(arr, class_idx):
    arr = tf.convert_to_tensor(arr)
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(arr, training=False)
        tape.watch(conv_out)
        loss = pred[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    w = tf.reduce_mean(grads, axis=(0,1,2))
    heat = tf.reduce_sum(conv_out[0]*w, axis=-1).numpy()
    heat = np.maximum(heat, 0)
    heat = heat/(heat.max()+1e-8)
    return cv2.resize(heat, (IMG_SIZE, IMG_SIZE))

# --------------------------------------------------------------- INTERFACE
fichier = st.file_uploader("Deposez une radiographie thoracique frontale (PA/AP)",
                           type=['png','jpg','jpeg'])

if fichier is not None:
    pil = Image.open(fichier)
    if not est_radio(pil):
        st.warning("⚠️ Cette image ne ressemble pas a une radiographie thoracique standard. "
                   "Le resultat serait peu fiable.")

    x, rgb = preparer(pil)
    probs = model.predict(x, verbose=0)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Radiographie")
        st.image(rgb, use_container_width=True)
    with col2:
        st.subheader("Zone d'attention (Grad-CAM)")
        try:
            top = int(np.argmax(probs))
            heat = gradcam(x, top)
            heat_c = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*heat), cv2.COLORMAP_JET),
                                  cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(rgb, 0.6, heat_c, 0.4, 0)
            st.image(overlay, use_container_width=True,
                     caption=f"Zone influencant : {FR[PATHOLOGIES[top]]}")
        except Exception as e:
            st.image(rgb, use_container_width=True)
            st.warning(f"Grad-CAM indisponible : {e}")

    ordre = sorted(zip(PATHOLOGIES, probs), key=lambda t: -t[1])
    positifs = [(n,p) for n,p in ordre if p >= SEUILS.get(n, 0.5)]

    st.subheader("Resultat")
    if positifs:
        st.markdown("**Pathologies signalees :**")
        for n, p in positifs:
            st.markdown(f"<div class='carte'><span class='pos'>{FR[n]}</span> "
                        f"- {p*100:.0f}% de confiance</div>", unsafe_allow_html=True)
    else:
        st.success("✅ Aucune anomalie detectee au-dessus du seuil de confiance.")

    st.markdown("**Top 5 probabilites :**")
    for n, p in ordre[:5]:
        st.write(f"{FR[n]} - {p*100:.1f}%")
        st.progress(float(p))

st.markdown("<div class='avert'>⚠️ <b>Avertissement medical.</b> Cet outil est une aide au "
            "pre-signalement a but pedagogique. Il n'est pas un dispositif medical valide et ne "
            "remplace en aucun cas l'interpretation d'un radiologue. Aucune decision clinique ne "
            "doit reposer sur ce resultat.</div>", unsafe_allow_html=True)
