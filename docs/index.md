# Segmented Image Blurring

## Abstract

To continue our understanding of deep learning with images, we wanted to create an application that can perform blurring on camera input, blurring unidentified objects in the background. We trained encoders and decoders to take an image and produce a mask. We then created a program that can use our model and blur the background of webcam images by creating a segmentation mask.

## Problem Statement

## Related Work

## Methodology

We split the project into 4 parts that could be worked on by each member of the group. The parts include pre-processing the data, creating and training a model, blurring an image given a mask, and creating an application that can take in camera input, using our model, and blur the background in real time.

### Pre-processing the data:

For our data, we used Pytorch’s Pascal VOC Dataset, training with year=2012. This produces an image of variable size as the data and a label that is a PNG. In order to train our model, we needed to resize every image. We ended up using cv2’s resize method.

### Creating the model:

Hello

### Blurring an image using a mask:

Hello

### Developing an app to blur in real-time:

Hello

## Experiments/Evaluation

## Results

## Examples

## Video




You can use the [editor on GitHub](https://github.com/JefferyT/Image-Segmentation/edit/main/docs/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/JefferyT/Image-Segmentation/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
