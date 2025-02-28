# Fashion-MNIST Generative Adversarial Network (GAN)
![__results___20_0](https://github.com/user-attachments/assets/d7dc1d80-204c-4671-a02a-2c50e771b589)


The **"Fashion-MNIST Generative Adversarial Network (GAN)"** project focuses on developing a GAN to generate realistic images of fashion items, such as clothing and accessories, by training on the Fashion-MNIST dataset. This dataset comprises 60,000 28x28 grayscale images of 10 fashion categories, including T-shirts, trousers, and dresses.

## Technical Approach

### 1. Data Preprocessing:
   - **Normalization:** The pixel values of the Fashion-MNIST images are scaled to a range of [-1, 1] to facilitate efficient training of the neural networks.
   - **Reshaping:** The images are reshaped to include a channel dimension, resulting in a shape of (28, 28, 1), which is compatible with convolutional neural networks (CNNs).

### 2. Model Architecture:
   - **Generator Network:**
     - **Input Layer:** Takes a random noise vector of a specified dimension (e.g., 100) as input.
     - **Dense Layer:** Transforms the input into a larger, reshaped tensor.
     - **Reshape Layer:** Reshapes the tensor into a format suitable for convolutional layers.
     - **Convolutional Transpose Layers:** Upsamples the tensor through several layers, progressively increasing spatial dimensions while decreasing depth, to generate a 28x28x1 image.
     - **Activation Functions:** Leaky ReLU is used in intermediate layers to introduce non-linearity, and Tanh is applied in the output layer to produce images with pixel values in the range [-1, 1].
   - **Discriminator Network:**
     - **Input Layer:** Accepts a 28x28x1 image.
     - **Convolutional Layers:** Extracts features from the image through several layers, progressively reducing spatial dimensions while increasing depth.
     - **Flatten Layer:** Converts the 3D feature maps into a 1D vector.
     - **Dense Layer:** Outputs a single scalar value representing the probability that the input image is real.
     - **Activation Function:** Leaky ReLU is used in intermediate layers, and a sigmoid function is applied in the output layer to produce a probability score between 0 and 1.

### 3. Training Process:
   - **Adversarial Training:** The generator and discriminator are trained simultaneously in an adversarial manner. The generator aims to produce images that the discriminator classifies as real, while the discriminator strives to distinguish between real and generated images.
   - **Loss Functions:**
     - **Discriminator Loss:** Measures the ability of the discriminator to correctly classify real and fake images.
     - **Generator Loss:** Measures the ability of the generator to produce images that the discriminator classifies as real.
   - **Optimization:** The Adam optimizer is used to minimize the loss functions, with separate optimizers for the generator and discriminator.

### 4. Evaluation and Results:
   - **Generated Images:** After training, the generator is used to produce new images from random noise vectors. These generated images are evaluated qualitatively to assess their realism and diversity.
   - **Training Visualization:** The project includes visualizations of the generator's output at various training stages to demonstrate the progression of image quality over time.

By implementing a GAN with convolutional architectures for both the generator and discriminator, this project effectively learns to generate realistic fashion images from random noise. The use of the Fashion-MNIST dataset provides a standardized benchmark for evaluating the performance of generative models in the fashion domain.
