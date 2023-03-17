# Carbonates prediction

This is an app that predicts carbonates content in soil samples based only on Fourier Transform Near-Infrared (FT-NIR) reflectance spectroscopy data.
We trained: 1) a Deep Neural Network (DNN) and 2) a Convolutional Neural Network (CNN) on the combined dataset of two near-Infrared (NIR) spectral libraries:
Kellogg Soil Survey Laboratory (KSSL) of the United States Department of Agriculture (USDA)<sup>1</sup>, a dataset of soil samples reflectance spectra collected nationwide,
and Land Use and Coverage Area Frame Survey (LUCAS) TopSoil (European Soil Library)<sup>2</sup> which contains soil sample absorbance spectra from all over the European Union,
and use them to predict carbonate content on never-before-seen soil samples. Soil samples in KSSL and in TopSoil spectral libraries were acquired in the spectral
region of visible–near infrared (Vis-NIR) (350-2500 nm), however in this study, only the NIR spectral region (1150-2500 nm) was utilized.

*<sub> 1. https://ncsslabdatamart.sc.egov.usda.gov/ </sub>* <br>
*<sub> 2. https://esdac.jrc.ec.europa.eu/content/lucas2015-topsoil-data  </sub>*
