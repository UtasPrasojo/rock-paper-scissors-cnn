import streamlit as st
from PIL import UnidentifiedImageError

from inference import load_artifacts, predict_image

THRESHOLD = 0.70  # di bawah ini tampilkan peringatan

st.set_page_config(page_title="Klasifikasi Rock-Paper-Scissors", page_icon="✊")


@st.cache_resource
def get_model():
    return load_artifacts()


st.title("✊✋✌️ Klasifikasi Rock-Paper-Scissors")
st.write("Upload gambar tangan (batu, kertas, atau gunting). Model CNN (MobileNetV2) akan menebak kelasnya.")

uploaded = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded:
    model, class_names, img_size = get_model()
    try:
        result = predict_image(uploaded, model, class_names, img_size)
    except UnidentifiedImageError:
        st.error("File tidak bisa dibaca sebagai gambar. Coba file jpg/png lain.")
        st.stop()

    left, right = st.columns(2)
    left.image(uploaded, caption="Gambar yang di-upload", width="stretch")

    with right:
        st.subheader("Hasil Prediksi")
        st.success(f"**{result['label'].upper()}** — confidence {result['confidence']:.1%}")
        if result["confidence"] < THRESHOLD:
            st.warning("Model kurang yakin, mungkin gambar bukan rock/paper/scissors.")
        st.write("Probabilitas tiap kelas:")
        for item in result["top_k"]:
            st.progress(item["confidence"], text=f"{item['label']}: {item['confidence']:.1%}")
