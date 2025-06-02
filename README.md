# Nerf 3D Change

## Description
<!-- Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors. -->
A codebase for experiments in detection of 3D Change in Neural Radiance Fields (NeRF).

The code can be used to generate NeRFs that take training data from two seperate zip files, and then generate multiple NeRF models and evaluate metrics such as 3D IoU for a set of models across a hyperparameter space. Hyperparamters for 3D Change and for NeRF can be set in a config file in the configs folder.

## Visuals
<!-- Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method. -->

## Getting started

First, you will need to generate zip files from Blender. The program functions by taking two zip files from blender of the same object, and generating a dataset with a mix of data from both sources.

### Install Blender

### Download Blender files - I have some hosted here!

### Install BlenderNeRF in Blender

https://github.com/maximeraafat/BlenderNeRF

### Generate COS Dataset for two scenes
Alternatively, use move two pretrained scene zips to your zip_files directory.

Steps/settings for properly configuring Blender

### Configuring the Config file

steps to set up your config file for use.

# Howdy!
To get started, I recommend installing all requirements in a conda environment. The requirements file contains everything that must be installed. Simply run the following commands to create one and install the required libraries:

```
conda env create -n "nerf3Dchange"
conda activate nerf3Dchange
pip install -r requirements.txt
```

# How to use this program:

The program main functionalities are in the jupyter .ipynb files:

**processing_training_data** - Use this file to process training data you generated from blender or omniverse code, so that it can be used in train_nerf.
train_nerf - Program to train a NeRF given rays (can be used for unzipped data you already have)
Testing - Use this to test PSNR in rendering novel views.
uncert_testing - Use this to test the rendering of entropy for an image.
MeshExtraction - Use this to generate a 3D mesh from your trained model.
visualize - Use this to visualize pose and ray data for one or many poses.

## Usage
<!-- Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README. -->

To run the experiment to reproduce results:
python train_nerf.py --conf configs/experiment1.conf

To evaluate models, run:
python evaluate.py

<!-- ## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README. -->

<!-- ## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser. -->

<!-- ## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers. -->
