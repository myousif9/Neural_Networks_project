# Neural_Networks_project

### Architectures for Denoising and deblurring Neural Networks
- Architecture 1 (DN-Net denosing network, DB-Net deblurring network & RestoreNET)
    - input layer: 128 x 128 x 3
    - 2 hidden layers: 
        - 8 feature maps
        - kernal size 5 x 5
        - stride = 1
        - sigmoid activation functions
    - output layer: 128 x 128 x 3
    - no pooling residual connections or other

- Architecture 2 (deep DN-Net denosing network, deep DB-Net deblurring network & deep RestoreNET)
    - input layer: 128 x 128 x 3
    - 4 hidden layers:
        - 24 feature maps
        - kernal size 5 x 5
        - stride = 1
        - sigmoid activation functions
    - output layer: 128 x 128 x 3
- loss function: mean squared error
- 100 epochs and stopped optimization if no improvement in validation sset after 5 consecutive evaluations