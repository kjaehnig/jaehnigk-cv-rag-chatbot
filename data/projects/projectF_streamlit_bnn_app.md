---
title: "Bayesian OCR Web-App"
excerpt: "<img src='/images/streamlit_bnn/OCR_demo_video.GIF'>"
collection: portfolio
---

{%include base_path%}


**STILL UNDER CONSTRUCTION**{: .notice--warning}


So this project has been ongoing for a bit and is really my first real implementation of one of my machine-learning models into the streamlit API as a web-app. I've been wanting to make something to really test my model and showcase its Bayesian capabilities in a sharable web app I can embed in websites or share as a link.

I expanded beyond doing just digit recognition, as I had done before with my Bayesian MNIST dense neural network. I chose the EMNIST dataset to provide the training and test images that would go into this expanded model. This dataset contains both the original MNIST digits, as well as English alphabet characters in both lower and upper case. This amounts to about 62 different labels in the entire dataset. I used the 'balanced' dataset, which has equal number of both training and testing samples for 47 labels: the 10 digits 0-9, and 37 letters. The 37 letters consist of all 26 upper case letters, and 11 lower case letters: a, b, d,e f, g, h, n, q, r, and t. The 15 lower case letters that are left out are those which have the same form in both upper and lower case, such as C (c), K (k), M (m) and so on. 

I chose this dataset because it let's me achieve a better performing model on all labels while shaving off the computational cost needed to classify more than just 10 digits. My previous MNIST Bayesian classifier model flattened and classified digits only on their pixel values. I will need to have more complex feature extraction to be able to identify all 47 different characters in the EMNIST dataset. Following the transfer-learning project I did previously with Pneumonia detection I will use a previously trained model in the Tensorflow applications module to perform feature extraction on the EMNIST images before running them through Bayesian dense layers to perform the final classifications. 

<iframe src="https://bnn-ocr-webapp.streamlit.app/?embedded=true" style="height:700px; width:100%;">
</iframe>

