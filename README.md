# Introduction
The purpose of the project was to represent real people's faces in a cartoon style. It can be used in the digital entertainment or AR field. Often, social media users spend time creating their own Avatar, and an application like this would make the process immediate. In video game development or animation film production, it could be used to generate specific references without necessarily relying on a concept artist. The approach used is auto-generative, utilizing two autoencoders: a real autoencoder for reconstructing real images and a cartoon autoencoder for reconstructing cartoon images. After training, model surgery was required to create a network that could take real images and return a cartoon version while retaining facial features.

The dataset consists of real face photos and digitally created cartoon face images. For cartoon images, a Google dataset was used, while for real images, a more elaborate dataset had to be constructed. The dataset did not use labels and included 87k cartoon training images, 8768 real training images, and 168 real test images. Initially, a subset of the datasets was used to find the right parameters and define the architecture. Then, the entire dataset was used to train the network. No data augmentation was done via code because the network was sensitive to color and rotation changes, and it was not wise to change these features manually.

Our goal was to reproduce the architecture presented in the paper *XGAN: Unsupervised Image-to-Image Translation for Many-to-Many Mappings* 
You can check it here: https://arxiv.org/abs/1711.05139

# Results

Unfortunately, we did not achieve a perfect and accurate conversion of every test image given as input. This was to be expected as the real dataset includes a very limited number of images (8768), some of which have poor lighting conditions. According to our observations, a dataset that is 10 times larger would be necessary to achieve more promising results.
