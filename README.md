# Fundus-Image-Denoising
Denoising a Fundus Image dataset using the U-Net CNN Architecture 

[Here](https://drive.google.com/drive/folders/1K-aI1LWjO3zbF-bjD0P9cMPWGsWiUKF-?usp=sharing) is the dataset which we have used for training the model (147 images)  
[Here](https://drive.google.com/drive/folders/1X5-iHF8hKE27rf3sJ943wtB4dPf2-C_d?usp=sharing) is the dataset we have used for testing the trained model (18 images)  

All the images have 3 color channels, i,e; they are RGB  

Training parameters: epochs = 50, batch size = 20, validation split = 20  

We first added some noise to the training dataset so we could train the model to predict the ground truth with the noised training data and we would test with the similarly noised testing dataset  

For the model we have used the U-Net Architecture, given below with the dimensions of the input image being (256, 256, 3)  

![u_net](https://miro.medium.com/max/680/1*TXfEPqTbFBPCbXYh2bstlA.png)  

## The training and validation metrics  

![history_plot](https://github.com/imjohnzakkam/Fundus-Image-Denoising/blob/main/comparision_images/history.png)  

We decided to sharpen the output image to enhance the visibilty of the optical nerves (arteries and veins) through a Computer Vision technique which is similar to the one given below: 

```py
def sharpen_img(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    image = cv2.filter2D(image, -1, kernel)
    return image
```

## Some of the resultant comparison images  

![comparison_image_0](https://github.com/imjohnzakkam/Fundus-Image-Denoising/blob/main/comparision_images/comparison-0.png)  
![comparison_image_1](https://github.com/imjohnzakkam/Fundus-Image-Denoising/blob/main/comparision_images/comparison-1.png)  
![comparison_image_10](https://github.com/imjohnzakkam/Fundus-Image-Denoising/blob/main/comparision_images/comparison-10.png)  
![comparison_image_7](https://github.com/imjohnzakkam/Fundus-Image-Denoising/blob/main/comparision_images/comparison-7.png)  

## Final Conclusions  
* With a 147 fundus image dataset we were able to produce a very close result of de-noising the images
* We are looking to improve the accuracy of this with a larger dataset and more improvised techniques  
* The code is compiled and put to a Google Colab .ipynb file [here](https://github.com/imjohnzakkam/Fundus-Image-Denoising/blob/main/fundus_image_denoise.ipynb)  
