"""
Drawing utils
"""

import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def draw_face_mesh(image: np.ndarray, results) -> np.ndarray:
    """
    draws face mesh

    Args:
        image (np.ndarray): _description_
        results (_type_): _description_

    Returns:
        np.ndarray: _description_
    """
    annotated_image = image.copy()
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        
    return annotated_image

