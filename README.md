# Time-Series Anomaly Detection for Non-Invasive Fetal ECG using Diffusion Transformer

## Overview
This repository provides a DiT-based NIFECG anomaly detection code developed as part of the ADVANCED GENERATIVE MODELS (STA8610) course project at Yonsei University Graduate School.

## Features
- Anomaly NIFECG dataset and generation script
- 1D DiT-based base diffusion model
- DDAD-based reconstruction module

## Installation
Clone this repository: `git clone https://github.com/Minjee-Seo/AnomalyDiT.git`

## Usage
1. Edit the `config.yaml` file in ddad_utils.
2. Run `python train.py` for training.
3. Run `python test.py` for sampling and evaluation.

## Contact
For any queries, please reach out to [Minjee Seo](mailto:islandz@yonsei.ac.kr) and [Seohyeon Jeong](mailto:jsh1021902@naver.com).
