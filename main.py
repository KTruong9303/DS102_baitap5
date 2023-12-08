import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

st.title('Handwirte Number Detection')

input = open('lrc_mnist.pkl', 'rb')
model = pkl.load(input)

st.header('up image')
image = st.file_uploader('choose!', type = (['png', 'jpg', 'jpeg']))

if image is not None:
  image = Image.open(image)
  st.image(image, caption='test iamge')

  if st.button('Predict'):
    image = image.resize((64, 1))
    feature_vector = np.array(image)
    label = str((model.predict(feature_vector))[0])

    st.header('Result')
    st.text(label)
    # st.text(class_list[label])
    
