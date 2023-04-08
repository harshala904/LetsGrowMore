#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python


# In[2]:


import cv2
import matplotlib.pyplot as plt
plt.style.use('seaborn')


# In[10]:


image = cv2.imread("C:/Users/a/Downloads/pexels-roshan-kamath-1661179.jpg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(6,6))
plt.imshow(image)
plt.axis("off")
plt.title("Original Image")
plt.show()


# #### Converting image to grey scale

# In[11]:


image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(6,6))
plt.imshow(image_gray,cmap="gray")
plt.axis("off")
plt.title("GrayScale Image")
plt.show()


# In[12]:


image_invert = cv2.bitwise_not(image_gray)
plt.figure(figsize=(6,6))
plt.imshow(image_invert,cmap="gray")
plt.axis("off")
plt.title("Inverted Image")
plt.show()


# In[13]:


image_smoothing = cv2.GaussianBlur(image_invert, (21, 21),sigmaX=0, sigmaY=0)
plt.figure(figsize=(6,6))
plt.imshow(image_smoothing,cmap="gray")
plt.axis("off")
plt.title("Smoothened Image")
plt.show()


# Converting image to pencil sketch

# In[15]:


sketch = cv2.divide(image_gray,255-image_smoothing, scale=255)
plt.figure(figsize=(6,6))
plt.imshow(sketch,cmap="gray")
plt.axis("off")
plt.title("Pencilsketch Image")
plt.show()


# ### Thank You

# In[ ]:




