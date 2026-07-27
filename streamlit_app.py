import os, urllib.request
import streamlit as st
import numpy as np, tensorflow as tf, cv2
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image

# ==========================================================================
#  A MODIFIER : mets ici l'adresse de TON modele sur Hugging Face.
#  Format : https://huggingface.co/TON-NOM/TON-REPO/resolve/main/NOM-DU-FICHIER
#  Exemple : https://huggingface.co/akert123/densenet-thorax/resolve/main/densenet121_14maladies_v2.keras
# ==========================================================================
MODEL_URL = "https://huggingface.co/akert123/densenet-thorax/resolve/main/densenet121_14maladies_v2.keras"
MODEL_PATH = "densenet121_14maladies_v2.keras"

st.set_page_config(page_title="Diagnostic thorax - 14 pathologies", layout="wide")

PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
               'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
               'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


@st.cache_resource
def charger_modele():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Telechargement du modele (une seule fois)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    dense_base = m.layers[0]
    inp = tf.keras.Input(shape=(224, 224, 3))
    feat = dense_base(inp)
    x = feat
    for layer in m.layers[1:]:
        x = layer(x)
    grad = tf.keras.Model(inp, [feat, x])
    return m, grad


model, grad_model = charger_modele()


def gradcam(arr, class_idx):
    arr = tf.convert_to_tensor(arr)
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(arr, training=False)
        tape.watch(conv_out)
        loss = pred[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    w = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv_out[0] * w, axis=-1).numpy()
    heat = np.maximum(heat, 0)
    heat = heat / (heat.max() + 1e-8)
    return cv2.resize(heat, (224, 224))


st.title("Diagnostic radiographie thorax - 14 pathologies")
st.write("Deposez une radiographie du thorax. Le modele affiche les maladies detectees "
         "et la zone sur laquelle il fonde sa decision (Grad-CAM).")
st.caption("Outil academique - ne remplace pas un avis medical.")

fichier = st.file_uploader("Radiographie du thorax", type=['png', 'jpg', 'jpeg'])

if fichier is not None:
    orig = np.array(Image.open(fichier).convert('RGB').resize((224, 224)))
    x = preprocess_input(np.expand_dims(orig.astype('float32'), 0))
    probs = model.predict(x, verbose=0)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Radiographie")
        st.image(orig, use_container_width=True)

    with col2:
        st.subheader("Zone d'attention (Grad-CAM)")
        try:
            top = int(np.argmax(probs))
            heat = gradcam(x, top)
            heat_c = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET),
                                  cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(orig, 0.6, heat_c, 0.4, 0)
            st.image(overlay, use_container_width=True)
        except Exception as e:
            st.image(orig, use_container_width=True)
            st.warning(f"Grad-CAM indisponible : {e}")

    st.subheader("Maladies detectees")
    ordre = sorted(zip(PATHOLOGIES, probs), key=lambda t: -t[1])
    for nom, p in ordre[:5]:
        st.write(f"**{nom}** - {p*100:.1f}%")
        st.progress(float(p))
