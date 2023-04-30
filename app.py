import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import numpy as np
import os
from streamlit_drawable_canvas import st_canvas
from io import StringIO
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionDepth2ImgPipeline

realtime_update = st.sidebar.button("Load the model")
selected_choice = st.sidebar.selectbox("Task", ("Generative", "Captured", "Drawing"))

if selected_choice == "Generative":

    if realtime_update:

        scheduler_text2image_base = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="scheduler")
        text2image = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base", scheduler=scheduler_text2image_base, torch_dtype=torch.float32)
        text2image = text2image.to("cuda" if torch.cuda.is_available() else "cpu")

        col1, col2 = st.columns(2)

        with col1:
            stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)


            initial_prompt = st.text_input('Write your prompt')
            img = text2image(initial_prompt, num_inference_steps=25).images[0]

            # st.image(img, caption=initial_prompt)
            canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.9)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color="#FFF",
            background_color="#000",
            background_image=img,
            update_streamlit=False,
            height=360,
            width = 360,
            drawing_mode="freedraw",
            point_display_radius= 0,
            key="canvas",
                    )
            ok_button = st.button('inpaint')  
        
            # st.image(canvas_result.image_data, caption="mask 2")    
            with col2:
            
                altered_prompt = st.text_input('Write your altered prompt')
                if ok_button:

                    pipe2 = StableDiffusionInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-inpainting",
                    torch_dtype=torch.float32,
                    )
                    pipe2.to("cpu")
                    image2 = pipe2(prompt=altered_prompt, image=img, mask_image=canvas_result.image_data, width=360, height=360,num_inference_steps=20).images[0]
                    st.image(image2, caption=altered_prompt)

elif selected_choice=="Captured":

    uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        with col1:
            image_file = Image.open(uploaded_file)
            image_file = image_file.resize((360,360), Image.LANCZOS)
            st.image(image_file)
            prompt = st.text_input("write prompt for depth to image")
            if st.button("Get Variations"):
                        
                depth2image = StableDiffusionDepth2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-depth",
                torch_dtype=torch.float32,
                ).to("cpu")

                
                
                image = depth2image(prompt=prompt, image=image_file,
                                    strength=0.8,num_inference_steps=10).images[0]
                
                
                with col2:
                    st.image(depth2image)
