import gradio as gr
import cv2
import pickle
import skimage
from skimage.feature import local_binary_pattern

clf = None
with open('classifier.pkl', 'rb') as f:
    clf = pickle.load(f)


def img2text(img):
  # print(img)
  # Resize the image to a specific width and height
  image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  resized_image = cv2.resize(image, (24, 24))

  # Compute the LBP feature vector
  lbp_feature_vector = local_binary_pattern(resized_image, 8, 1, method="uniform")

  # Print the feature vector
  # print(lbp_feature_vector)

  flattened_arr = lbp_feature_vector.reshape(-1)
  # print(flattened_arr)

  y_pred = clf.predict([flattened_arr])
  if y_pred[0] == 0: 
    return 'face'
  else:
    return 'non-face'



import gradio as gr 

# gr.Interface(txt2img, gr.Image(), gr.Text(), title = 'Stable Diffusion 2.0 Colab with Gradio UI').launch(share = True, debug = True)

iface = gr.Interface(img2text, gr.Image(), gr.Text(), title = 'Face Detector: Local Binary Pattern method, Support Vector Machine algorithm')
iface.launch()




# file_path = 'images/Copy of 35.jpg'

# # Load the image
# image = cv2.imread(file_path)
# print(image.shape)

# # Resize the image to a specific width and height
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# resized_image = cv2.resize(image, (24, 24))

# lbp_feature_vector = local_binary_pattern(resized_image, 8, 1, method="uniform")

# flattened_arr = lbp_feature_vector.reshape(-1)


# y_pred = clf.predict([flattened_arr])
# print(y_pred)
