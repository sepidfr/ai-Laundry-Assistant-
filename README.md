# ai-Laundry-Assistant

This project implements an end-to-end intelligent laundry-sorting system using a large-scale real-world fashion dataset.
The model takes a single clothing image as input and predicts:

Color group

Fabric type

Recommended washing cycle

Clothing vs. non-clothing (IS_CLOTHING gate)

The system is built using a 4-head multitask ConvNeXt-Tiny architecture, trained on a curated subset of ~16,800 labeled samples from the H&M Fashion Dataset on Kaggle:
https://www.kaggle.com/code/gpreda/h-m-eda-and-prediction/input


# About the Notebook Files

The full training notebook is large due to the dataset size and preprocessing pipeline.
To keep the repository lightweight and avoid GitHub upload limits, the notebook is included as a ZIP file.

You can extract the ZIP and run the notebook directly without any modifications.

![ai](https://github.com/user-attachments/assets/f2d44904-a3e5-4a7f-84c0-ae093887c339)
