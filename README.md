## 1) Pré-requisitos

- Python 3.8+ instalado.
- Acesso ao MongoDB (Atlas ou local) e a string de conexão (`MONGO_URI`).
- Terminal (PowerShell, cmd ou bash).

## 2) Dependências — instalação

No terminal, dentro da pasta do projeto, rode:

```bash
pip install streamlit pymongo face-recognition opencv-python numpy pillow

```

> Windows: se der erro ao instalar dlib/face-recognition, você pode precisar do CMake instalado e no PATH. Instale o CMake (marque “Add CMake to PATH”) e depois rode:
> 
> 
> ```bash
> pip install CMake
> pip install dlib
> pip install face_recognition
> 
> ```
> 

## 3) Preparar a aplicação

1. Abra o arquivo `app.py`.
2. Verifique a variável `MONGO_URI` e substitua pela sua string de conexão do MongoDB (ou ajuste conforme seu ambiente).
3. Salve o arquivo.

## 4) Executar a aplicação

No terminal, entre na pasta que contém `app.py` e rode:

```bash
python -m streamlit run app.py

```

Ao rodar, o Streamlit abrirá automaticamente no navegador em `http://localhost:8501`. Se não abrir, copie o link mostrado no terminal e cole no navegador.

## 5) Como usar a aplicação (passo a passo)

### Menu lateral

A aplicação tem 2 opções no menu lateral:

### A) **Inserir imagens no MongoDB**

1. Clique em *Inserir imagens no MongoDB*.
2. Clique em *Selecionar imagens para salvar no banco* e escolha uma ou várias imagens (jpg/png/jpeg).
3. Clique em **Salvar no MongoDB**.
4. A aplicação tentará extrair um *embedding* (vetor facial) para cada imagem. Apenas imagens com rosto detectado serão salvas.
5. Você verá uma mensagem de sucesso informando quantas imagens foram salvas.

> Campos salvos no banco: nome_arquivo, imagem_base64, embedding (vetor numérico).
> 

### B) **Comparar com minha foto**

1. Clique em *Comparar com minha foto*.
2. Use o botão para *Enviar uma foto* (jpg/png/jpeg) — essa será a sua imagem de referência.
3. A imagem será mostrada na tela. A aplicação extrai o embedding da sua imagem.
4. O app buscará todas as imagens no MongoDB que tenham embeddings e calculará distância Euclidiana entre vetores.
5. O app exibirá:
    - **Mais parecido** — imagem do banco com menor distância.
    - **Mais diferente** — imagem com maior distância.
6. Você verá o nome do arquivo e a **distância** numérica (menor = mais parecido).

## 6) Interpretação das distâncias

- A métrica usada é distância Euclidiana entre embeddings. **Menor distância = maior similaridade**.
- Não existe um valor universal perfeito; para `face_recognition` costuma-se considerar ~0.6 como limite (valores menores indicam alta semelhança), mas isso varia com o modelo e dataset.

## 7) Boas práticas para fotos

- Use fotos de rosto de frente, sem óculos escuros ou filtros exagerados.
- Iluminação boa e rosto centralizado aumentam acurácia.
- Evite imagens muito pequenas ou de baixa resolução.

## 8) Problemas comuns e soluções rápidas

- **Erro `No module named 'face_recognition'`**
    
    Instale: `pip install face_recognition` — se falhar, instale CMake e dlib antes (ver seção 2).
    
- **Comando `streamlit` não encontrado**
    
    Rode: `python -m streamlit run app.py` ou adicione a pasta `...AppData\Roaming\Python\PythonXYZ\Scripts` ao PATH do Windows.
    
- **Nenhum rosto detectado na imagem enviada**
    
    Tente outra foto mais clara e de frente.
    
- **Banco vazio / nenhuma imagem encontrada**
    
    Verifique se as imagens foram salvas (coleção `imagens` no DB) ou faça upload via a aba de Inserir imagens.
    
- **Conexão com MongoDB falha**
    
    Confira a `MONGO_URI` e se o IP/cliente têm permissão (no Atlas, libere seu IP ou permita 0.0.0.0/0 enquanto testa).
    

## 9) Segurança e privacidade

- Não compartilhe imagens sensíveis ou dados pessoais em ambientes públicos.
- Considere remover imagens do banco quando não forem mais necessárias.

## 10) Fechar a aplicação

No terminal onde rodou o Streamlit, pressione `Ctrl+C` para encerrar o servidor.

---

## Exemplo rápido de fluxo (resumido)

1. Ajuste `MONGO_URI` no `app.py`.
2. `pip install ...` (dependências).
3. `python -m streamlit run app.py`.
4. Menu → Inserir imagens → Fazer upload → Salvar.
5. Menu → Comparar com minha foto → Fazer upload → Comparar → Ver mais parecido / mais diferente.
