import streamlit as st
import numpy as np
from pymongo import MongoClient
from PIL import Image
import io, base64
import cv2

# ==========================
# CONFIG MONGO
# ==========================
MONGO_URI = "mongodb+srv://cemngarcia_db_user:ocHltIGr04QD3zCB@cluster0.jqoofvy.mongodb.net/?appName=Cluster0"
DB_NAME = "atividade14"
COLLECTION_NAME = "imagens"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# ==========================
# FACE DETECTOR (OpenCV)
# ==========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==========================
# FUNÇÕES
# ==========================

def imagem_para_base64(imagem_bytes):
    return base64.b64encode(imagem_bytes).decode("utf-8")

def base64_para_imagem(image_base64):
    return Image.open(io.BytesIO(base64.b64decode(image_base64)))

def bytes_para_cv2(imagem_bytes):
    nparr = np.frombuffer(imagem_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def gerar_embedding(imagem_bytes):
    img = bytes_para_cv2(imagem_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, (100, 100))
    embedding = face.flatten() / 255

    return embedding


# ==========================
# INTERFACE
# ==========================

st.set_page_config(page_title="Atividade 14 - Reconhecimento Facial", layout="centered")
st.title("🧠 Comparação Facial - MongoDB + Streamlit (OpenCV)")

menu = st.sidebar.selectbox(
    "Menu",
    ["Inserir imagens no banco", "Comparar com minha foto"]
)

# ==========================
# 1 - INSERIR
# ==========================
if menu == "Inserir imagens no banco":

    st.header("📥 Salvar imagens no MongoDB")

    imagens = st.file_uploader(
        "Selecione imagens",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button("Salvar"):

        if imagens:
            total = 0

            for img in imagens:
                img_bytes = img.getvalue()
                img_base64 = imagem_para_base64(img_bytes)

                embedding = gerar_embedding(img_bytes)

                if embedding is not None:
                    doc = {
                        "nome_arquivo": img.name,
                        "imagem_base64": img_base64,
                        "embedding": embedding.tolist()
                    }

                    collection.insert_one(doc)
                    total += 1

            st.success(f"✅ {total} imagens salvas no banco!")

        else:
            st.warning("⚠️ Nenhuma imagem selecionada")


# ==========================
# 2 - COMPARAR
# ==========================
elif menu == "Comparar com minha foto":

    st.header("📸 Comparar sua imagem")

    metodo = st.radio(
        "Envio da imagem:",
        ["Upload", "Câmera"]
    )

    if metodo == "Upload":
        minha_imagem = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])
    else:
        minha_imagem = st.camera_input("Tire a foto")

    if minha_imagem:

        img_bytes = minha_imagem.getvalue()
        imagem = Image.open(io.BytesIO(img_bytes))
        st.image(imagem, width=250)

        embedding_usuario = gerar_embedding(img_bytes)

        if embedding_usuario is None:
            st.error("❌ Nenhum rosto detectado")
        else:
            docs = list(collection.find())

            if len(docs) == 0:
                st.warning("Sem imagens no banco")
            else:
                resultados = []

                for doc in docs:
                    emb = np.array(doc["embedding"])
                    distancia = np.linalg.norm(embedding_usuario - emb)

                    resultados.append({
                        "nome": doc["nome_arquivo"],
                        "imagem": doc["imagem_base64"],
                        "distancia": distancia
                    })

                mais_parecido = min(resultados, key=lambda x: x["distancia"])
                mais_diferente = max(resultados, key=lambda x: x["distancia"])

                st.subheader("✅ Mais parecido")
                st.image(base64_para_imagem(mais_parecido["imagem"]), width=200)
                st.write("Nome:", mais_parecido["nome"])
                st.write("Distância:", round(mais_parecido["distancia"], 4))

                st.subheader("❌ Mais diferente")
                st.image(base64_para_imagem(mais_diferente["imagem"]), width=200)
                st.write("Nome:", mais_diferente["nome"])
                st.write("Distância:", round(mais_diferente["distancia"], 4))
