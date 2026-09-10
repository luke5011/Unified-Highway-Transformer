
# Unified Highway Transformer (UHT)

[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.33510313-blue)](https://doi.org/10.6084/m9.figshare.33510313)

Official repository for the Unified Highway Transformer (UHT). The UHT introduces an O(1) parallel memory bus to solve Latent Amnesia and PreNorm Dilution in deep networks. It eliminates gradient stagnation, with successful stress tests conducted up to 180 layers. 

---

Quick Start & Reproducibility
1. Installation
Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Dataset Preparation
Download and tokenize the WikiText-103 dataset using the standard GPT-2 encoder:

```bash
python prepare.py
```
3. Rapid Evaluation (10-Second Verification)
You can instantly verify the 120-layer terminal validation loss (4.31) and perplexity (74.88) claimed in the paper without running a 50-epoch training cycle:

Download the pre-trained weights (uht_120L_lean.pt) from the Releases tab on the right side of this repository page.

Place the .pt file in the main repository directory.

Run the evaluation command:

```bash
python train.py --eval_only --checkpoint uht_120L_lean.pt
```
4. Full Training Run
To initiate a full training cycle from scratch on your own hardware (automatically logs gradient deciles and variance tracking to a CSV):

```bash
python train.py
```
## Visualization of Convergence

<img width="1200" height="800" alt="120L_gradient_comparison" src="https://github.com/user-attachments/assets/c6f4ab12-594c-46d3-a66f-fe4eb09148ba" />

**Above:** 120-Layer UHT vs. Baseline Pre-LN Transformer. The UHT maintains dynamic gradient flow deep in the network, while the standard architecture stagnates near zero.

<img width="1200" height="400" alt="180L_gradient_comparison" src="https://github.com/user-attachments/assets/ea9925f7-03f4-4fda-ae89-bc9dfa5859ef" />

**Above:** 180-Layer UHT stress test, verifying sustained gradient activity and stability at extreme architectural depths.

---

## Read the Paper

The complete manuscript detailing the methodology, Amorphous Neural Network diagnostic proxies, and empirical test results is available directly in this repository:
[📄 Read the UHT Paper (PDF)](Unified%20Highway%20Transformer%20Paper_3.pdf)

---

## Citation

If you use this architecture or diagnostic methodology in your research, please cite:

```bibtex
@article{simon2026uht,
  title={Designing Better Transformers: Using Amorphous Network Proxies to Engineer the Unified Highway Transformer},
  author={Simon, Anthony Luke},
  year={2026},
  publisher={Figshare},
  doi={10.6084/m9.figshare.33510313},
  url={https://doi.org/10.6084/m9.figshare.33510313}
}
```
