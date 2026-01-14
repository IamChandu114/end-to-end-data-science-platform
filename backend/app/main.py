from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os

app = FastAPI(title="End-to-End Data Scientist Platform")

# -------------------------------------------------
# PATH SETUP (THIS MATCHES YOUR FOLDER STRUCTURE)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR -> backend/

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Safety checks (very important for beginners)
if not os.path.exists(FRONTEND_DIR):
    raise RuntimeError(f"frontend folder not found: {FRONTEND_DIR}")

if not os.path.exists(STATIC_DIR):
    raise RuntimeError(f"static folder not found: {STATIC_DIR}")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------------------------------
# CSV ANALYSIS
# -------------------------------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    result = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "preview": df.head().to_dict()
    }

    return JSONResponse(content=result)

# -------------------------------------------------
# PREDICTION (SIMPLE DEMO MODEL)
# -------------------------------------------------
@app.post("/predict")
async def predict(data: dict):
    try:
        age = float(data["age"])
        salary = float(data["salary"])
        experience = float(data["experience"])
    except:
        return {"error": "Invalid input"}

    score = age * 0.3 + experience * 2 + salary * 0.0001
    prediction = "High" if score > 10 else "Low"

    return {"prediction": prediction}
