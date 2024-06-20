from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
import os

# Inicializar o aplicativo FastAPI
app = FastAPI()

# Carregar o modelo treinado
model_path = "model/model.h5"
model = load_model(model_path)

# Mapeamento das classes
class_indices = {
    0: "TRASEIRA",
    1: "QRCODELACRE",
    2: "PLACAS",
    3: "MOTOR",
    4: "HODOMETRO",
    5: "ETIQUETA",
    6: "DIANTEIRA",
    7: "CHASSI"
}

# Mapeamento de perguntas para classes
question_map = {
    "é uma traseira de carro?": 0,
    "é um qr code do lacre?": 1,
    "é uma placa?": 2,
    "é um motor?": 3,
    "é um hodômetro?": 4,
    "é uma etiqueta?": 5,
    "é uma dianteira?": 6,
    "é um chassi?": 7
}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), question: str = Form(...)):
    print(f"Pergunta recebida: '{question}'")
    if question.lower() not in question_map:
        return JSONResponse(content={"error": "Pergunta não reconhecida."}, status_code=400)

    # Salvar o arquivo enviado
    temp_file = f"static/{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Carregar e pré-processar a imagem
    img = image.load_img(temp_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Fazer a previsão
    prediction = model.predict(img_array)
    predicted_probabilities = prediction[0]

    expected_class_index = question_map[question.lower()]
    confidence_percentage = predicted_probabilities[expected_class_index] * 100

    # Remover o arquivo temporário
    os.remove(temp_file)

    return JSONResponse(content={"confidence": f"{confidence_percentage:.2f}%"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
