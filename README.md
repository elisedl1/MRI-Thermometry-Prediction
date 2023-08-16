# BrainSeg-Setup
**Segmentation Code:**
  <br />-"segmentation_all.ipynb"
  <br />-The root directory contains all "LP-xxxx" files. This must include anatomicalProbesEye and temperatureData.
  <br />-references/explanations are included in code at the end of each major function
  <br /> -extractor.py in repo <u>MUST REPLACE THE IMPORTED VERSION</u>
  <br />       -this can be done by copy/pasting

**Resampling/Cropping Code:**
<br />  -"resample_crop.ipynb"
  <br />-requires MTLE files as root directory
  <br />-remove final "break" in bottom cell to run with all patients
  <br />-references/explanations are included in code at the end of each major function

**Utils Code:**
<br /> -"utils.ipynb"
<br /> -removes t2w MRIs (as well as any other specified ones) using variable called bad_list
<br /> -combines segmentation and anatomical MRIs probesEye pngs for use in training

**Segmentation + Resampling Pipeline Visualization:**

<br />![pipeline](https://github.com/elisedl1/BrainSeg-Setup/assets/95655831/187ebf73-a665-480c-a13b-35315f908b29)
