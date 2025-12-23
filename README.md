# anime_reality_classifier
Overview
This pipeline automatically extracts images from .tar files, classifies them as either "anime-style" or "realistic" using a fine-tuned EfficientNet-B0 model, and outputs organized results with comprehensive statistics.

Model Performance:

Validation Accuracy: 92.9%

Test Accuracy: 88.4%

Inference Speed: ~100 images/second (GPU)
Basic Usage
After including the zip.tar in the repository, please run the pipeline with python run.py zip1.tar --output-dir ./results 
