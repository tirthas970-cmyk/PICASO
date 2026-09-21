<h1 align="center">PICASO</h1>
<p align="center"><b>P</b>hysics <b>I</b>nformed <b>C</b>ross <b>A</b>ttention <b>S</b>olar <b>O</b>ptimizer</p>
<div align="center">
  <a href="https://github.com/tirthas970-cmyk">tirthas970-cmyk</a>
</div>

<div align="center">
  <a href="https://github.com/Alexanderiscool1">Alexanderiscool1</a>
</div>
<br><br>
Insert image later

## Overview
We built a CNN to predict M and X-Class flares using mutli-model data. Using AI Explainabillity techniques, we attempted to solve the **Black Box** problem in order to see why the model is making its decisions. 
Data consists of magnetograms and physical qualities that correlate with solar flare activity

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

Create table

