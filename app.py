import matplotlib.pyplot as plt
import streamlit as st
from deepOstinato.preprocessing.short_time_fourier_transform import STFT
from deepOstinato.preprocessing.plot_spectogram import plot_spectrogram

st.set_page_config(
page_title="DeepOstinato", # => the title of the tab
page_icon="🎼", #icon on the tab
layout="centered", # wide or centered
initial_sidebar_state="collapsed") # collapsed or auto

st.title("""DeepOstinato 🎼""")

st.sidebar.title("""
     Navigation

    """)
with st.sidebar:
 st.markdown('[Home](#deepostinato)', unsafe_allow_html=True)
 st.markdown('[About](#about)', unsafe_allow_html=True)
 st.markdown('[Model](#model)', unsafe_allow_html=True)
 st.markdown('[Github](https://github.com/RaoulConstantine/deepOstinato)', unsafe_allow_html=True)


st.subheader("""
             You're here because you want to see what your music file would look like, as a Spectogram.
            """)

with st.expander("What's a Spectogram? 🤔"):
    #an expander to display the definition of a spectogram to the user when clicked.
     st.write(""" """) #write the definition here

st.text('Go ahead, try it yourself! upload a WAV or MP3 below')


st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose an MP3 OR WAV")

if uploaded_file is not None:
    st.write('you uploaded the file')


st.header('Model', 'model')
st.markdown("""What is Lorem Ipsum?
        Lorem Ipsum is simply dummy text of the printing and typesetting industry.
        Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
        when an unknown printer took a galley of type and scrambled it to make a type specimen book.
        It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged.
        It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages,
        and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
        It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout.
        The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here',
        making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy.
        Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).""")


st.header('About', 'about')
st.markdown("""DeepOstinato uses a DCGAN model developed by students at Le Wagon
            as their final project""")
