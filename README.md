# Machine Learning Engineering Assignment

This repository contains a deep learning–based solution for **action recognition** and **temporal localization** on a video stream. The assignment follows a series of steps outlined by the Matrice Hiring Team, from dataset preparation and model design to training, evaluation, and (ideally) deployment.

---

## Table of Contents
1. [Overview of the Tasks](#overview-of-the-tasks)
2. [Dataset and Pre-Processing](#dataset-and-pre-processing)
3. [Research on State-of-the-Art (SOTA)](#research-on-sota)
4. [Model Design and Architecture](#model-design-and-architecture)
5. [Training and Evaluation](#training-and-evaluation)
6. [Results and Observations](#results-and-observations)
7. [Deployment Considerations](#deployment-considerations)
8. [Code Organization](#code-organization)
9. [Pylint Score](#pylint-score)

---

## Overview of the Tasks

1. **Download/Use a Temporal Action Localization Dataset**  
   - The assignment suggests ActivityNet, but due to its large size (4+ hours just to load), I used **pre-extracted features from THUMOS14**.

2. **Create a Small Subset of the Dataset**  
   - Worked with `.npy` features instead of raw videos to handle limited resources and time.

3. **Research Top 3 SOTA Papers on Temporal Action Localization**  
   - Investigated how BMN, BSN, and SSN handle boundary modeling, proposal generation, and structured segments.

4. **Design a Small Custom Model**  
   - Built a lightweight 1D CNN (`TemporalLocalizationModel`) for window-level classification, balancing simplicity and quick training.

5. **Train the Model and Evaluate Performance**  
   - Trained on the THUMOS14 subset for 10 epochs; computed mean Average Precision (mAP).

6. **Compare with Models Trained on ActivityNet**  
   - Direct comparison not feasible due to dataset differences. The code structure, however, mirrors standard temporal localization pipelines.

7. **Install a Video Streaming Server and Demonstrate Real-Time Inference**  
   - Not fully completed because `.npy` features do not include raw video frames for a real-time camera feed.

---

## Dataset and Pre-Processing

- **Dataset**: THUMOS14 Pre-Extracted Features  
  - Chosen due to time constraints and to avoid large data loading issues.  
  - Contains `.npy` files for validation and test splits, each shaped `(num_frames, feature_dim)`.

- **Subset Creation**:  
  - Filtered out missing feature files, performed train-validation-test splits.

**Note**: Initially attempted ActivityNet, but the loading time on the GPU server exceeded 4 hours, so I pivoted to THUMOS14 features.

---

## Research on SOTA

- **BSN (Boundary Sensitive Network)**: Focuses on boundary detection.  
- **BMN (Boundary Matching Network)**: Uses boundary matching for proposal generation.  
- **SSN (Structured Segment Network)**: Segments each action instance into structured parts.

**Key Points**:  
- Advanced boundary modeling can greatly improve temporal localization.  
- My simplified approach is a 1D CNN that doesn’t implement boundary refinement.

---

## Model Design and Architecture

- **TemporalLocalizationModel**  
  - A 1D convolutional layer (kernel_size=3), AdaptiveMaxPool1d, and a fully connected layer.  
  - Input shape: `[batch_size, window_size, feature_dim]`  
  - Output: Class scores for each window.

- **Why This Design**:  
  - Lightweight for quick training.  
  - Inspired by the local temporal context from SOTA, but simplified due to time constraints.

---

## Training and Evaluation

1. **Training**  
   - **PyTorch** with Adam (`lr=1e-4`) and cross-entropy loss.  
   - Automatic Mixed Precision (AMP) with `GradScaler`.  
   - Trained for 10 epochs on a small subset of `.npy` features.

2. **Evaluation**  
   - Computed **mAP** at IoU threshold = 0.5.  
   - Post-processed window-level predictions with `post_process_predictions`.  
   - Final mAP score: **0.0005**, indicating limited performance likely due to minimal hyperparameter tuning and data constraints.

---

## Results and Observations

- **mAP Score**: **0.0005**  
  - Very low, indicating the model didn’t generalize well.  
  - Possible reasons:
    - Simplified approach without boundary refinement.
    - Limited data and minimal hyperparameter tuning.
    - Pre-extracted features may not align perfectly with the model’s needs.

- **Comparison to ActivityNet**:  
  - Not directly comparable due to dataset differences and time constraints, but the pipeline is standard for temporal localization tasks.

---

## Deployment Considerations

- **Video Streaming Server**  
  - References in `deploy.py` show how one might implement real-time streaming with Flask/SocketIO, but no raw videos were available to demonstrate a live camera feed.
- **Exporting the Model**  
  - The model is saved in `.pth` format and can be adapted to a standard camera feed if raw frames are available.

---

## Code Organization

- **`main.py`**  
  - Loads the dataset, trains the model, runs inference, and computes mAP.  
- **`evaluation.py`**  
  - Provides utility functions like `compute_map` and `post_process_predictions`.  
- **`model.py`**  
  - Defines the `TemporalLocalizationModel`.  
- **`deploy.py`** (Optional)  
  - Illustrates a possible real-time deployment script if raw frames were accessible.  
- **`thumos_anno_action.json`** and **`thumos_features/`**  
  - Annotation file and pre-extracted feature files for THUMOS14.

---

## Pylint Score

- The project’s code has been refactored to follow PEP8 standards, proper docstrings, and snake_case naming conventions.  
- **Pylint score**: Currently **above 9**, indicating good code quality despite some warnings about too many local variables or instance attributes, which could be refactored further.

---

Thank you for reviewing this project. Although the final mAP score is quite low, the pipeline demonstrates a complete approach for temporal localization using pre-extracted features, following a modular code structure that could be optimized with more time and resources.
