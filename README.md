# Neural Style Transfer

Neural Style Transfer (NST) is an exciting application of deep learning that combines the content of one image with the style of another, creating visually stunning results. This project demonstrates a PyTorch implementation of Neural Style Transfer, along with simple instructions for running the code and reproducing the results.

## How Neural Style Transfer Works

Neural Style Transfer (NST) takes two input images: a **content image**, which provides the structure, and a **style image**, which provides the artistic style. The goal is to generate a new image that preserves the content of the content image while adopting the style of the style image. This project provides a Python-based implementation of NST using PyTorch, making it easy to experiment with different images and styles.

### More detailed explanation

Neural Style Transfer (NST) leverages the feature extraction capabilities of Convolutional Neural Networks (CNNs) to blend the content structure of one image with the artistic style of another. Here are the key steps involved in the NST process:

##### 1. Feature Extraction via Pre-Trained CNNs: 

NST uses a pre-trained CNN model, such as VGG-19, to compute hierarchical feature representations of the content and style images. Early layers capture low-level features like edges and textures, while deeper layers capture high-level semantic features.

##### 2. Content and Style Representations:

Content Representation: The output of a specific intermediate layer of the CNN serves as the content feature map. This feature map encapsulates the spatial arrangement and structural information of the content image.

Style Representation: Style is represented using the Gram matrix, a statistical measure of feature correlations within each layer. The Gram matrix encodes textures and patterns by capturing spatial dependencies between feature maps.

##### 3. Loss Function:

Content Loss: Defined as the squared difference between the content features of the generated image and the content image, ensuring the generated image retains the structural elements of the content image.

Style Loss: Quantifies the difference between the Gram matrices of the generated image and the style image across multiple layers, ensuring the generated image emulates the textures and patterns of the style image.

##### 4. Optimization:
The NST process starts with an initialized image (which in this repository is the content image, other options are random noise or the style image) and iteratively updates its pixel values via gradient descent (This repo uses the Adam optimizer). The total loss function combines content and style losses, weighted by hyperparameters. The optimization adjusts the image to minimize the total loss, balancing content preservation and style transfer.

## Results

Below are some examples of Neural Style Transfer results, showcasing the fusion of various content and style images:

<table>
  <thead>
    <tr>
      <th>Content Image</th>
      <th>Style Image</th>
      <th>Result Image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="data/content/buildings.jpg" width="300"></td>
      <td><img src="data/styles/starry_night.jpg" width="300"></td>
      <td><img src="data/examples/buildings_starry_night.jpg" width="300"></td>
    </tr>
    <tr>
      <td><img src="data/content/city.jpg" width="300"></td>
      <td><img src="data/styles/wheat_fields.jpg" width="300"></td>
      <td><img src="data/examples/city_wheat_fields.jpg" width="300"></td>
    </tr>
    <tr>
      <td><img src="data/content/batad_rice_terraces.jpg" width="300"></td>
      <td><img src="data/styles/scream.jpg" width="300"></td>
      <td><img src="data/examples/batad_rice_terraces_scream.jpg" width="300"></td>
    </tr>
    <tr>
      <td><img src="data/content/batad_rice_terraces.jpg" width="300"></td>
      <td><img src="data/styles/watercolor.jpg" width="300"></td>
      <td><img src="data/examples/batad_rice_terraces_watercolor.jpg" width="300"></td>
    </tr>
    <tr>
      <td><img src="data/content/devils_bridge.jpg" width="300"></td>
      <td><img src="data/styles/the_night_cafe.jpg" width="300"></td>
      <td><img src="data/examples/devils_bridge_the_night_cafe.jpg" width="300"></td>
    </tr>
    <tr>
      <td><img src="data/content/earthrise.jpg" width="300"></td>
      <td><img src="data/styles/scream.jpg" width="300"></td>
      <td><img src="data/examples/earthrise_scream.jpg" width="300"></td>
    </tr>
    <tr>
      <td><img src="data/content/golden_gate_driving.jpg" width="300"></td>
      <td><img src="data/styles/edtaonisl.jpg" width="300"></td>
      <td><img src="data/examples/golden_gate_driving_edtaonisl.jpg" width="300"></td>
    </tr>
    <tr>
      <td><img src="data/content/golden_gate.jpg" width="300"></td>
      <td><img src="data/styles/scream.jpg" width="300"></td>
      <td><img src="data/examples/golden_gate_scream.jpg" width="300"></td>
    </tr>
    <tr>
      <td><img src="data/content/golden_gate.jpg" width="300"></td>
      <td><img src="data/styles/pencil_sketch.jpg" width="300"></td>
      <td><img src="data/examples/golden_gate_pencil_sketch.jpg" width="300"></td>
    </tr>
    <tr>
      <td><img src="data/content/mona_lisa.jpg" width="300"></td>
      <td><img src="data/styles/edtaonisl.jpg" width="300"></td>
      <td><img src="data/examples/mona_lisa_edtaonisl.jpg" width="300"></td>
    </tr>
    <tr>
      <td><img src="data/content/taj_mahal.jpg" width="300"></td>
      <td><img src="data/styles/starry_night.jpg" width="300"></td>
      <td><img src="data/examples/taj_mahal_starry_night_smaller.jpg" width="300"></td>
    </tr>
    <tr>
      <td><img src="data/content/taj_mahal.jpg" width="300"></td>
      <td><img src="data/styles/wheat_fields.jpg" width="300"></td>
      <td><img src="data/examples/taj_mahal_wheat_fields.jpg" width="300"></td>
    </tr>
  </tbody>
</table>


You can find more examples in the [`examples`](/data/examples/) folder.


## Installation

1. Clone the repository
```
git clone https://github.com/lassebaerlandstrand/Neural-Style-Transfer.git
cd Neural-Style-Transfer
```
2. Make sure you have `uv` installed (faster alternative to pip). Follow uv's installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/)
3. You are now ready to run the project!

## How to Run the Project
1. Prepare the content and style images, and put them in the corresponding folders in the [`data`](/data) directory.
2. Change the file names in the [`main.py`](/src/main.py) file to match the content and style images you want to use. You can also adjust the hyperparameters and optimization settings in the file.
3. Run the [`main.py`](/src/main.py) file with this command (if this is the first time running this command, `uv` will create a virtual environment and install all dependencies automatically).
```
uv run src/main.py
```
4. The generated image will be saved in the [`generated`](/data/generated/) folder.

## Acknowledgements and References
- The images used in this project are sourced from Wikipedia. Some images are color adjusted in GIMP to have more clear colors.
- The implementation is inspired by the seminal paper ["A Neural Algorithm of Artistic Style" by Gatys et al. (2015)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).
- Video series by Aleksa Gordic on Neural Style Transfer on [Youtube](https://www.youtube.com/watch?v=S78LQebx6jo&list=PLBoQnSflObcmbfshq9oNs41vODgXG-608).
- Pre-trained VGG-19 weights are used, available via the PyTorch model zoo.