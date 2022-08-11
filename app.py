#Kerakli kutubxonalar
import streamlit as st
from fastai.vision.all import *
import pathlib
import matplotlib.pyplot as plt
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath

#nomi
st.title("Ovqatlarnni klassifikatsiya qiluvchi model")

#rasmni joylash
file=st.file_uploader("Rasm yuklash", type=["png","jpeg","gif","svg",'jpg'])
if file:
    st.image(file)

    #pil convert
    img=PILImage.create(file)

    #model
    model=load_learner("food_model.pkl")

    #predict
    prediction, pred_id, probs=model.predict(img)
    st.success(f'bashorat: {prediction}')
    st.info(f'Ehtimollik :{probs[pred_id]*100:.1f}')
    #plots
    fig=plt.plot(x=probs*100,y=model.dls.vocab)
    st.plotly_chart(fig)
