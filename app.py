import streamlit as st
import numpy as np
from pymongo import MongoClient
from PIL import Image
import io
import base64
import mediapipe as mp
import cv2

# ==========================
# CONFIGURAÇÃO DO MONGO
# ==========================
MONGO_URI = "mongodb+srv://cemngarcia_db_user:ocHltIGr04QD3zCB@cluster0.jqoofvy.mongodb.net/?appName=Cluster0"
DB_NAME = "atividade14"
COLLECTION_NAME = "imagens"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# ==========================
# CONFIGURAÇÃO MEDIAPIPE
# ==========================
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ==========================
# FUNÇÕES
# ==========================

def imagem_para_base64(imagem_bytes):
    return base64.b64encode(imagem_bytes).decode("utf-8")

def base64_para_imagem(image_base64):
    return Image.open(io.BytesIO(base64.b64decode(image_base64)))

def preprocessar_imagem(imagem_bytes):
    nparr = np.frombuffer(imagem_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_image

def gerar_embedding(imagem_bytes):
    image = preprocessar_imagem(imagem_bytes)
    results = face_detection.process(image)

    if not results.detections:
        return None

    # Pegando a primeira face
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box

    h, w, _ = image.shape
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    w_box = int(bbox.width * w)
    h_box = int(bbox.height * h)

    face = image[y:y+h_box, x:x+w_box]

    if face.size == 0:
        return None

    # Padronizando rosto
    face = cv2.resize(face, (100, 100))

    # Gerando embedding simples (normalizado)
    embedding = face.flatten() / 255.0

    return embedding


# ==========================
# INTERFACE
# ==========================

st.set_page_config(page_title="Atividade 14 - Reconhecimento Facial", layout="centered")
st.title("🧠 Atividade 14 - MongoDB + Streamlit + MediaPipe")

menu = st.sidebar.selectbox(
    "Menu",
    ["Inserir imagens no MongoDB", "Comparar com minha foto"]
)

# ==========================
# 1 - INSERIR IMAGENS
# ==========================
if menu == "Inserir imagens no MongoDB":

    st.header("📥 Inserir imagens no MongoDB")

    imagens = st.file_uploader(
        "Selecione imagens",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button("Salvar no banco"):

        if imagens:
            total = 0

            for img in imagens:
                img_bytes = img.getvalue()
                img_base64 = imagem_para_base64(img_bytes)

                embedding = gerar_embedding(img_bytes)

                if embedding is not None:
                    documento = {
                        "nome_arquivo": img.name,
                        "imagem_base64": img_base64,
                        "embedding": embedding.tolist()
                    }

                    collection.insert_one(documento)
                    total += 1

            st.success(f"✅ {total} imagens salvas no MongoDB!")

        else:
            st.warning("⚠️ Nenhuma imagem selecionada.")


# ==========================
# 2 - COMPARAR COM MINHA FOTO
# ==========================
elif menu == "Comparar com minha foto":

    st.header("📸 Descobrir a imagem mais parecida e mais diferente")

    metodo = st.radio(
        "Como deseja enviar sua foto?",
        ["Enviar imagem", "Tirar foto pela câmera"]
    )

    if metodo == "Enviar imagem":
        minha_imagem = st.file_uploader("Envie sua foto", type=["jpg", "jpeg", "png"])
    else:
        minha_imagem = st.camera_input("Tire sua foto")

    if minha_imagem:

        img_bytes = minha_imagem.getvalue()
        imagem = Image.open(io.BytesIO(img_bytes))

        st.image(imagem, caption="Sua foto", width=250)

        embedding_usuario = gerar_embedding(img_bytes)

        if embedding_usuario is None:
            st.error("❌ Nenhum rosto detectado na imagem")
        else:
            documentos = list(collection.find())

            if len(documentos) == 0:
                st.warning("⚠️ Nenhuma imagem no banco.")
            else:
                distancias = []

                for doc in documentos:
                    embedding_banco = np.array(doc["embedding"])
                    distancia = np.linalg.norm(embedding_usuario - embedding_banco)

                    distancias.append({
                        "nome": doc["nome_arquivo"],
                        "distancia": distancia,
                        "imagem_base64": doc["imagem_base64"]
                    })

                mais_parecido = min(distancias, key=lambda x: x["distancia"])
                mais_diferente = max(distancias, key=lambda x: x["distancia"])

                st.subheader("✅ Mais parecido")
                img1 = base64_para_imagem(mais_parecido["imagem_base64"])
                st.image(img1, width=200)
                st.write(f"Nome: {mais_parecido['nome']}")
                st.write(f"Distância: {mais_parecido['distancia']:.4f}")

                st.subheader("❌ Mais diferente")
                img2 = base64_para_imagem(mais_diferente["imagem_base64"])
                st.image(img2, width=200)
                st.write(f"Nome: {mais_diferente['nome']}")
                st.write(f"Distância: {mais_diferente['distancia']:.4f}")
