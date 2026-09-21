<h1 align="center">PICASO</h1>
<p align="center"><b>P</b>hysics <b>I</b>nformed <b>C</b>ross <b>A</b>ttention <b>S</b>olar <b>O</b>ptimizer</p>
<div align="center">
  <a href="https://github.com/tirthas970-cmyk">tirthas970-cmyk</a>
</div>

<div align="center">
  <a href="https://github.com/Alexanderiscool1">Alexanderiscool1</a>
</div>
<br><br>

## Overview
We built a CNN to predict M and X-Class flares using mutli-model data. Using AI Explainabillity techniques, we attempted to solve the **Black Box** problem in order to see why the model is making its decisions. 
Data consists of magnetograms and physical qualities that correlate with solar flare activity

Methodology:
<img width="1403" height="821" alt="image" src="https://github.com/user-attachments/assets/d9042b36-a004-4a95-87b5-525e8fe7b080" />


## Introduction
Solar flares are large outbursts on the sun that send bursts of energy, light, and fast-moving particles into space.
They are grouped by power:
* A, B, C: Are small flares, and are barely noticeable on Earth
* M-Class: Medium flares that can cause brief radio blackouts near Earth’s poles
* X-Class: The strongest explosions in the solar system, and can cause planet-wide radio blackouts and radiation storms

### The Problem:
* Forecasting and predicting large flares (i.e., M and X-class) are important so that we can ensure safety and protect technology
  * Solar flares disrupt Earth’s communication, navigation, and power system 
       * Due to intense bursts of electromagnetic radiation
  * Endangers astronauts from radiation

Traditional methods struggle with rapid and complex magnetic dynamics, and existing ML models lack multi-model fusion and physical explainabillity.

## Data
As said before, magnetograms, which show the magnetic field of the sun, were the spatial data for our model.

The tabular data was fetched from the SHARP database, a solar physics dataset published by NASA derived from their Helioseismic and Magnetic Imager instrument (HMI) aboard the Solar Dynamics Observatory (SDO).

The 9 chosen parameters:

| Parameter | Description | Unit |
| :--- | :--- | :--- |
| R_VALUE | Complexity along the polarity inversion line (the "danger zone" for flares) | Mx |
| TOTPOT | Total magnetic energy stored in the region | erg |
| MEANPOT | How concentrated energy is (energy per area) | erg/cm³ |
| USFLUX | Magnetic Strength of a region | Mx |
| MEANSHR | Average shear of flare-prone fields | Radians |
| SHRGT45 | How much of the region is in strong shear (>45°) | arcsec² |
| TOTUSJH | Total current helicity | mG² |
| MEANJZH | Normalized helicity (twist per area) | G² |
| AREA_ACR | Size of the active region; larger complex regions flare more | arcsec² |

We uploaded all of our data into a public Hugging Face Dataset:

Link: https://huggingface.co/datasets/X-FlareNet/solar-flare-magnetogram-shards/tree/main

We fetched the magnetograms from JSCOC, and because the size of each file was too large, we packaged them into .tar shards in the form of chunks. Each chunk is typically 2-4 months of data.

Magnetograms were collected from years 2012, 2014, 2017, 2023, and 2024 at 4 hour intervals. Each tabular parameter was bounded to each magnetograms through their respective AR numbers, time stamps, ID, etc.

Our Hugging Face dataset also contains our splits and the meta-table.

Empirical Data Statistics

| Total Mangetograms | 60785
| --- | --- |
| Quiet Regions | 58453
| M-Class Flares| 2074
| X-Class Flares| 258

### Data Processing

We used the Cylindrical Equal Area (CEA) projection version of the magnetograms. CEA is used to ensure that pixel sizes correspond to equal physical areas on the Sun. It works by wrapping a cylinder around a sphere and then projecting the surface outwards. The grid lines for latitude are spaced proportional to the sine of latitude.

Our images were padded into a uniform 512 x 512, preventing ratio distortion, and preserving spatial coordinates.

To mitigate the severe class imbalance we implemented focal loss and class weights:

#### 1. Inverse Log Class Weights

For a given class distribution count vector $N$, the inverse frequency for class $i$ is calculated as:

$$\text{InvFreq}_i = \frac{\sum_{j} N_j}{N_i}$$

The log-smoothed weight with a $1.0$ offset is defined as:

$$w_i = \ln(\text{InvFreq}_i) + 1.0$$

The final class balance coefficient $\alpha_i$ is normalized relative to the first class ($w_0$):

$$\alpha_i = \frac{w_i}{w_0}$$

---

#### 2. Balanced Focal Loss

For a single sample with target class $t$, let $p_t$ be the model's estimated probability for that ground-truth class. The standard cross-entropy loss is:

$$\text{CE}(p_t) = -\ln(p_t)$$

Applying the focusing parameter $\gamma$ and the normalized class weight $\alpha_t$, the balanced focal loss for that sample is:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \ln(p_t)$$

The final batch loss optimized during the forward pass is the mean over $N$ samples:

$$\mathcal{L} = \frac{1}{N} \sum_{n=1}^{N} \alpha_{t_n} (1 - p_{t_n})^\gamma \text{CE}(p_{t_n})$$

---

#### 3. Hierarchical Loss Formulation

The multi-head loss combines the binary head (Quiet vs. Flare) and the severity head (M-Class vs. X-Class) using an indicator mask $\mathbb{I}$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{binary}} + \mathbb{I}_{(\text{targets} > 0)} \cdot \mathcal{L}_{\text{severity}}$$

## Model Architecture 

<img width="1800" height="675" alt="image" src="https://github.com/user-attachments/assets/20b160af-8d17-4eda-86ec-a8ae788ba8d5" />


### Feature Extraction & Tokenization 
* **Spatial Processing**: Stripped the pooling/classification layers from a standard ResNet-18 to capture the raw spatial tensor of shape (B, 512, H’, W’)
* Flattened the 2D grid into sequence-first image tokens of shape (B, H’*W’, 512)
* Implemented 2D positional encoding in order to preserve spatial features
  * Split the 512 channels into independent vertical (256) and horizontal (256) sine/cosine grids
* Processed 9 physical parameters through 9 separate MLP for each instead of one concentrated vector
* Each independent parameter is projected into 512-dimensional embedding, creating a stacked sequence of shape (B, 9, 512)

### Asymmetric Fusion & Dual Head Inference
* Implemented a Cross-Attention engine:
  * Tabular tokens act as queries, and spatial tokens act as keys/values
  * This answers *Which unique region of the magnetogram matters most for these specific physical measurements?*
* Computed attention matrix is added directly back to the original tabular physics token (Residual Fusion)
  * Guarantees the network remembers input parameters after spatial contextualization 
* Transformer Encoder Layer: Forces the 9 parameters to talk to each other only after they have fully processed the spatial image conext 
* Dual Linear Heads: The final averaged vector (from Global Pooling) is fed into the parallel MLPs to simultaneously output:
  * Head 1: Flare vs. No Flare (Binary)
  * Head 2: M-Class vs. X-Class
* Model also outputs attention weights, which is where the model is focusing on

## Results:

Our best model produced these results:

| Metric | Value |
| :--- | :--- |
| **True Negatives** | 5,036 |
| **False Positives** | 1,929 |
| **False Negatives** | 26 |
| **True Positives** | 229 |
| **F1 Score** | 0.1898 |
| **True Skill Statistic (TSS)** | 0.6211 |
| **Heidke Skill Score (HSS)** | 0.1352 |

**Confusion Matrix on Test Set**:
<img width="565" height="482" alt="image" src="https://github.com/user-attachments/assets/5e426572-ae30-4aa7-ad27-f2fb4b6a9752" />

### Interpreting The Results:
* Low F1 & HSS scores due to the high FP rates and low precision
  * Most likely from the class imbalance as flares only made it up about 3% of our total data 
  * To minimize being penalized heavily, the model may have guessed ‘yes’ and risk false positive 
  * Another interesting theory we had was that some magnetograms may look dangerous and erupt as a flare, but many stressed regions can rotate out of view without actually erupting
     * Known as the **loaded gun** effect in solar forecasting
* The TSS score was high as TSS calculates the difference between the true positive rate and the false positive rate
  * This means that TSS is independent of the class imbalance 
  * Our .62 TSS score shows that our model is able to differentiate a quiet and flare event
  * This is a production quality TSS in solar forecasting

## Model Explainability

### Saliency Maps
* Grad-CAM and specifically the attention map shows that the model targeted active-region green zones, which shows high magnetic strength, but also suffered from boundary shortcut learning near edge padding
  * Edge = high contrast between mangetogram and empty space, leading to more attraction of CNN filters
  * The model did not focus highly on the **Polarity Inversion Line**, which is where most flares happen
* **Spatial Ablation**: Blurring edge pixels actually increased confidence slightly
   * This may have been because it removed boundary noise (i.e., from the edges), allowing the model to focus on the Polarity Inversion Line and other green zones

<figure>
  <img src="https://github.com/user-attachments/assets/4dc5791f-c3ca-493b-a826-e4cda2952d12" alt="Alternate description text">
  <figcaption align="center"><b></b> The first column shows the magnetogram. The second column shows the attention map. The third column shows the Grad-CAM. The fourth column shows the magnetogram after the spatial ablation.</figcaption>
</figure>
<br><br>

### Feature Attribution 
* Modality Ablation: 
  * Removing images entirely caused a 6.5% drop in confidence,
  * Removing the entire tabular data caused a 36.8% drop in confidence
  * This shows that the model relied heavily on tabular data and confirming that images provided vital spatial context
* SHAP plot proved that current helicity (TOTUSJH, R_VALUE, MEANJZH) and magnetic shear (MEANSHR) drove flare prediction 

<figure>
  <img src="https://github.com/user-attachments/assets/3e0f573c-687b-446a-8e4d-955364999f6a" alt="Alternate description text">
  <figcaption align="center"><b></b>Evaluation set contains mostly non-flaring regions evaluated against a 30-flare baseline, meaning when the model encounters a region with low magnetic current and helicity, it pushes its prediction away from a flare. This is accurate to real solar physics.</figcaption>
</figure>
<br><br>

#### Key Takeaway: The model is physically grounded and context-aware but is constrained by computer vision biases

## Conclusion & Discussion
* The model was able to successfully differentiate flares vs. no flares, as well as figure out potentially dangerous magnetograms
* SHAP & Ablation analysis confirmed the model’s decisions were driven by real solar physics indicators like current helicity and magnetic shear, but Grad-CAM revealed a tendency for boundary shortcut learning
* Future work:
   * Explicitly guiding future architectures to focus on the Polarity Inversion Line through a loss function
   * Temporal Data: Allows the model to track the evolution of a specific active region (e.g., 24 hour time span)
   * Run on multiple seeds: Our final model was a specific epoch of it. We did did run multiple times, and got similar results, but we didn't do a full on mutli-seed experiment.
* Real-World Application: Enhances automated space weather forecasting to protect infrastructure and public safety 
* Unlike traditional models that analyze data like magnetograms in isolation, our architecture dynamically uses tabular and spatial data, forcing the model to learn mathematically and physically consistent precursors to solar flares 

> Note: This project is considered complete.


