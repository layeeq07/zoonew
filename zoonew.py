import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
df = pd.read_csv('zoo.data')

x = df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']]
y = df['type']

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)
lr.fit(xtrain,ytrain)

from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def example(a:int,b:int,c:int,d:int,e:int,f:int,g:int,h:int,i:int,j:int,k:int,l:int,m:int,n:int,o:int,p:int):
    pred = lr.predict([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p]])
    return str(pred)
    