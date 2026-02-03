#run server using command: fastapi dev apimain.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from testlogich import heavy_calculation

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # måste matcha port som webbsidan körs på
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputData(BaseModel):
    number: int


@app.post("/run")
def run_python(data: InputData):
    print(f"run_python with data {data}")
    result = heavy_calculation(data.number)
    print(f"Got result: {result}")
    return {"result": result}

@app.post("/prediction")
def run_prediction():
    print("running prediction")
    #kod kommer
    return {}