from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = joblib.load("house_price_model.pkl")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
def predict(
    request: Request,
    area: float = Form(...),
    bedrooms: int = Form(...),
    bathrooms: int = Form(...)
):
    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms]],
        columns=["Area", "Bedrooms", "Bathrooms"]
    )

    prediction = model.predict(input_data)

    price = round(prediction[0], 2)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": price}
    )