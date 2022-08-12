from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pickle
import pandas as pd

class Details(BaseModel):
    name: str
    age: int

class SendingData(BaseModel):
    YearsAtCompany: float
    EmployeeSatisfaction: float
    Position: str
    Salary: int

app = FastAPI()
with open('model.pkl','rb') as f:
    model = pickle.load(f)
    
@app.get('/')
async def welcome():
    return {'message': 'This is the homepage of testing ml model via api. '}

@app.get('/api/{name}')
def test(name: str):
    return {'message': f'Hello {name}'}

@app.post('/api/')
def api(data: Details):
    return {'message': f'Hello my name is {data.name} and my age is {data.age}'}

@app.post('/api/prediction/')
def prediction(data: SendingData):

    df =  pd.DataFrame([data.dict().values()],columns=data.dict().keys())
    yhat = model.predict(df)
    return {
        'prediction' : int(yhat)
    }
if __name__ == '__main__':
    uvicorn.run(app,port=8080,debug=True)
