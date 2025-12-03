import streamlit as st
import face_recognition
import numpy as np
from pymongo import MongoClient
from PIL import Image
import io
import base64
import os

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
# FUNÇÕES
# ==========================
def imagem_para_base64(imagem_bytes):
    return base64.b64encode(imagem_bytes).decode("utf-8")


def base64_para_imagem(image_base64):
    return Image.open(io.BytesIO(base64.b64decode(image_base64)))


def extrair_embedding(imagem_bytes):
    imagem = face_recognition.load_image_file(io.BytesIO(imagem_bytes))
    locations = face_recognition.face_locations(imagem)

    if len(locations) == 0:
        return None

    encoding = face_recognition.face_encodings(imagem, locations)[0]
    return encoding


# ==========================
# INTERFACE
# ==========================
st.title("Atividade 14 - MongoDB + Streamlit + Reconhecimento Facial")

menu = st.sidebar.selectbox(
    "Menu",
    ["Inserir imagens no MongoDB", "Comparar com minha foto"]
)

# ==========================
# 1 - INSERIR IMAGENS NO BANCO
# ==========================
if menu == "Inserir imagens no MongoDB":

    st.header("Inserir imagens no MongoDB")

    imagens = st.file_uploader(
        "Selecione imagens para salvar no banco",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if st.button("Salvar no MongoDB"):

        if imagens:

            total = 0

            for img in imagens:

                img_bytes = img.getvalue()
                img_base64 = imagem_para_base64(img_bytes)

                encoding = extrair_embedding(img_bytes)

                if encoding is not None:
                    documento = {
                        "nome_arquivo": img.name,
                        "imagem_base64": img_base64,
                        "embedding": encoding.tolist()
                    }

                    collection.insert_one(documento)
                    total += 1

            st.success(f"{total} imagens foram salvas com sucesso no MongoDB!")

        else:
            st.warning("Nenhuma imagem selecionada.")


# ==========================
# 2 - COMPARAR MINHA FOTO
# ==========================
elif menu == "Comparar com minha foto":

    st.header("Descobrir a imagem mais parecida e mais diferente")

    minha_imagem = st.file_uploader(
        "Envie uma foto sua",
        type=["jpg", "png", "jpeg"]
    )

    if minha_imagem:

        img_bytes = minha_imagem.getvalue()
        imagem = Image.open(io.BytesIO(img_bytes))

        st.image(imagem, caption="Sua foto", width=250)

        encoding_usuario = extrair_embedding(img_bytes)

        if encoding_usuario is None:
            st.error("Nenhum rosto detectado na imagem enviada!")

        else:
            documentos = list(collection.find())

            if len(documentos) == 0:
                st.warning("O banco não possui imagens.")
            else:

                distancias = []

                for doc in documentos:

                    embedding_banco = np.array(doc["embedding"])
                    distancia = np.linalg.norm(encoding_usuario - embedding_banco)

                    distancias.append({
                        "nome": doc["nome_arquivo"],
                        "distancia": distancia,
                        "imagem_base64": doc["imagem_base64"]
                    })

                mais_parecido = min(distancias, key=lambda x: x["distancia"])
                mais_diferente = max(distancias, key=lambda x: x["distancia"])

                # ==========================
                # RESULTADOS
                # ==========================
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
