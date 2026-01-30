from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from testlogich import heavy_calculation

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # exakt din frontend
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputData(BaseModel):
    number: int


@app.post("/run")
def run_python(data: InputData):
    result = heavy_calculation(data.number)
    return {"result": result}

