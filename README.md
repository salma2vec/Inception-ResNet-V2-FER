# Inception-ResNet-V2: Face Emotion Recognition (FER)

**Reference Paper**: [Inception-v4, Inception-ResNet, and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261v1.pdf)

## Overview

This PyTorch model is based on the Inception-ResNet-V2 architecture and is designed for facial emotion recognition. 
- This model has a total of **26 high-level blocks** and can classify upto **1001** different classes of images. 
- It has a complete depth of 164 layers.
- The model has a depth of 164 layers with **input-size** = `3 x 299 x 299`.

## Implementation Details

### Model Design

- The model's blocks are explicitly defined, specifying `in_channels` and `out_channels` for each layer, enhancing the visual flow of image processing.
- A custom `LambdaScale` layer is introduced to scale the residuals, addressing the issue of early training convergence in later layers.
- Batch normalization is applied to ensure regularization.

  ![image](https://user-images.githubusercontent.com/70141886/161433653-8d46d38e-39ab-4bc0-b335-ef374b45b469.png)
  <p align=center> Layer design - Overview </p>
  
### Parameters

- Loss function: `torch.nn.CrossEntropyLoss()`
- Optimizer: `torch.optim.Adam(amsgrad=True)`
- Learning rate scheduler: `torch.optim.lr_scheduler.ReduceLROnPlateau(mode='min', factor=0.2, threshold=0.01, patience=5)`

### Training

- `prefetch_generator.BackgroundGenerator` is utilized to improve computational efficiency by pre-loading the next mini-batch during training.
- The `state_dict` of each epoch is stored in the `resnet-v2-epochs` directory, created if it doesn't exist.
- The model is designed to run on a CUDA GPU, falling back to a CPU if no GPU is detected.
- Parallelization is not implemented to maintain code readability and ease of implementation.
- Training results can be interactively monitored using `TensorBoard`, with logs stored in the `/runs` directory.
- A benchmark of 00:30:03 hours was achieved on a system with an NVIDIA GTX 1650Ti 4GB, Intel i7-10750H, 16GB RAM, and an SSD during one epoch of training.

### Dataset and Pre-processing

- The model is trained on the Face-Expression-Recognition-Dataset from [jonathanoheix](https://www.kaggle.com/jonathanoheix) on Kaggle, containing 28,821 images of 7 emotion classes: 'Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', and 'Surprise'.
- Preprocessing includes resizing using `torchvision.transforms.Resize()`, converting to tensors with `torchvision.transforms.ToTensor()`, and normalization with `torchvision.transforms.Normalize()`.
- `torch.utils.data.DataLoader()` is used for efficient loading and processing of random mini-batches.

## Set-Up

You can choose to run the project using either the Jupyter notebook or the provided scripts in the `Scripts` folder of this repository.

#### Jupyter Notebook

1. Run the cells in order, adjusting parameters as needed. You can increase the number of `epochs` with available hardware.
2. Helper functions within the notebook cells allow generating predictions for images using the trained models.

#### Scripts

1. Ensure you have the required dependencies installed by running `pip install -r requirements.txt --no-index`.
2. Modify parameters in `train.py`, which contains the code for training the model defined in `resnet_model.py`.
3. If using VS Code, deploy a TensorBoard session by clicking on `Launch TensorBoard session` above the `Tensorboard` import in the file. Otherwise, follow the steps provided in [Using TensorBoard with PyTorch](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html).

## Credits

- **Paper**: *Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning*
- **Authors**: *Christian Szegedy, Sergey Ioffe, and Vincent Vanhoucke*
- **Image Dataset Source**: [Face Expression Recognition Dataset](https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset)
