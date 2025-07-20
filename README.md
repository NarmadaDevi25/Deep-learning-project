# Deep-learning-project
*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: PULI NARMADA DEVI

*INTERN ID*:CT06DH143

*DOMAIN*: DATA SCIENCE

*DURATION*: 6 WEEKS

*MENTOR*: NEELA SANTOSH

This my second task of developing a deep learning model for image classification using pytorch for this task i used google colab First, I installed PyTorch and other necessary libraries, such as torchvision and matplotlib. You can install PyTorch using: pip install torch torchvision

Building an image classification model in PyTorch typically involves several key steps that ensure the effective loading and preprocessing of your image data. 

1. **Load and Preprocess the Image Data**: 

   Begin by utilizing `torchvision.datasets.ImageFolder`, which is particularly useful for datasets organized in a directory structure where each class is represented by a separate folder. This function allows you to easily load images and automatically assigns labels based on the folder names. 
   Next, implementing `torchvision.transforms` for comprehensive preprocessing of your images. This stage is crucial for improving model performance and involves various transformations, such as:
   - **Resizing**: Adjusting the dimensions of images to maintain consistency across your dataset. This is important because neural networks often require inputs of the same size.
   - **Center Cropping**: This technique helps in focusing on the central part of the image, which often contains the main subject, and can be particularly useful in scenarios like face recognition.
   - **Normalization**: Normalizing pixel values is essential for stabilizing the training process. This typically involves scaling pixel values to a range of [0, 1] or standardizing them to zero mean and unit variance, depending on the network’s requirements.
   Additionally, makeing  sure to convert image data into tensors using the `transforms.ToTensor()` function, which is necessary for PyTorch models to process the image data effectively. Combining these preprocessing steps helps ensure that your images are in the optimal format for training, ultimately leading to better model accuracy and generalization to unseen data.Use torch.utils.data.DataLoader to efficiently load and iterate through the dataset in batches.
Augment the training dataset to improve model generalization. Examples include random horizontal flips, rotations, and crops.
Next, coming to  Convolutional Neural Network (CNN) is a specialized type of neural network, particularly well-suited for processing data with a grid-like topology, such as images. To implement a CNN architecture, you can define a class that inherits from `torch.nn.Module`, which is a foundational building block in PyTorch for creating neural networks.
1. **Convolutional Layers**: `nn.Conv2d` is used for applying convolution operations, which help in detecting features in the input image by scanning it with learnable filters.
2. **Pooling Layers**: `nn.MaxPool2d` is employed to down-sample the spatial dimensions of the feature maps, effectively reducing the computational load and helping to introduce invariance to small translations in the input.
3. **Fully Connected Layers**: At the end of the network, you might include `nn.Linear` layers for classification tasks. These layers combine the features extracted by the convolutional layers to make predictions about the input data.
Additionally, activation functions are often applied after the convolutional layers, with ReLU (Rectified Linear Unit) being a popular choice. This introduces non-linearity into the model, enabling it to learn complex patterns in the data.train the Model:
Iterate through the training dataset in batches (epochs).
Perform forward pass, calculate loss, backpropagate gradients, and update model weights using the optimizer.
Consider using a learning rate scheduler to adjust the learning rate over time.
Track and plot training loss and accuracy to monitor progress and identify overfitting or underfitting.
**After the CNN we train the Model:**
1. **Data Preparation:** Begin by splitting the training dataset into smaller batches, often referred to as epochs, to facilitate efficient processing and enhance model learning. This batching process ensures that the model can generalize better and avoids memory overload.
2. **Model Training Loop:** For each epoch, iterate through these batches and perform a forward pass through the model to generate predictions. This involves inputting the batch data and obtaining the output from the model, which will then be used for loss calculation.
3. **Loss Calculation:** After the forward pass, compute the loss, which quantifies the difference between the predicted outputs and the true labels. This critical step helps in determining how well the model is performing and guides the adjustment of model parameters.
4. **Backpropagation:** Utilize backpropagation to calculate the gradients of the loss function concerning the model weights. This process involves applying the chain rule of calculus to propagate the error backward through the network, enabling the determination of how to adjust the weights to minimize the loss.
5. **Weights Update:** Use an optimization algorithm (such as Adam, SGD, or RMSprop) to update the model weights based on the calculated gradients. It’s essential to fine-tune the learning rate in this step, as it determines the size of the update applied to the model weights. 
6. **Learning Rate Scheduling:** Consider implementing a learning rate scheduler to adjust the learning rate dynamically throughout training. This can help in stabilizing the training process, allowing for larger updates when the model is far from convergence and smaller updates as it nears a solution.
Evaluate the Trained Model:
Evaluate the model on a separate test dataset to measure its performance on unseen data.
Use metrics like accuracy to quantify the percentage of correct predictions.

output:

<img width="851" height="825" alt="Image" src="https://github.com/user-attachments/assets/c9e1ba25-e033-4d92-aefd-20cf9785ce12" />
