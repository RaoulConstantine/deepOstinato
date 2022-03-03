import streamlit as st

st.markdown("""DeepOstinato
            You're here because you want to see what your music file would look like,
            as a Spectogram.
            """)

with st.expander("What's a Spectogram?"):
    #an expander to display the definition of a spectogram to the user when clicked.
     st.write(""" """) #write the definition here

st.text('Go ahead, try it yourself! upload a WAV or MP3 below')


st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose an MP3 OR WAV", type=['mp3, wav'])

if uploaded_file is not None:
    Spectogram = ''
