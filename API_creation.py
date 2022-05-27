# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:56:04 2022

@author: CSU5KOR
"""

#https://anderfernandez.com/en/blog/how-to-create-api-python/
from fastapi import FastAPI

#app=from fastapi import FastAPI

app=FastAPI()

@app.get("/my-first-api")

def hello(name=None):
    if name is None:
        text='Hello'
    else:
        text= 'Hello '+ name + '!'
    return text
