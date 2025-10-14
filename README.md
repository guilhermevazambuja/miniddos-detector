# MiniDDoS Detector

A lightweight pipeline for detecting Distributed Denial of Service (DDoS) attacks using behavioral clustering and
dimensionality reduction techniques. This project leverages the CICIDS2017 dataset to explore timing-based features and
visualize network flow patterns through Singular Value Decomposition (SVD).

## About the Dataset: CICIDS2017

The [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset was created to address the limitations of older
intrusion detection datasets. It includes both benign and attack traffic captured in realistic network conditions, with
labeled flows derived from PCAPs using CICFlowMeter. The dataset reflects modern attack types and user behavior across
protocols like HTTP, HTTPS, FTP, SSH, and email.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/miniddos-detector.git
cd miniddos-detector
```

```bash
pip install -r requirements.txt
````
