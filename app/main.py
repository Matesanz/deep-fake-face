"""
MAIN MODULE
"""

from typing import Union
import streamlit as st
import numpy as np
from PIL import Image
import av
import cv2


from pose_estimation.solutions import face_mesh
from pose_estimation.drawing import draw_face_mesh
from streamlit_webrtc import webrtc_streamer


class VideoProcessor:
    """
    class used by streamer to process real time frames
    """
    def recv(self, frame):
        """
        process frame
        """
        image = frame.to_ndarray(format="bgr24")
        results = face_mesh.process(image)
        annotated_image = draw_face_mesh(image, results)
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


def select_solution() -> str:
    """
    Display selectbox with available models         

    Returns:
        str: Model key
    """

    selected_label = st.selectbox(
        'Select a Model',
        list(labels.keys())
    )
    st.write('You selected:', selected_label)
    return labels[selected_label]


def upload_image() -> Union[np.ndarray, None]:
    """
    Shows uploader of images in streamlit

    Returns:
        Union[np.ndarray, None]: Returns image if 
            uploaded otherwise returns None
    """

    image_bytes = st.file_uploader(
        "Upload an Image to Process", ["jpg", "jpeg", "png"]
        )

    if image_bytes:
        return np.asarray(Image.open(image_bytes))


if __name__ == "__main__":

    st.title("MediaPipe Showroom")
    st.text("Select a model and upload an image and see the result!")
    st.markdown("""---""")


    with st.container():

        webrtc_streamer("cam", video_processor_factory=VideoProcessor)
