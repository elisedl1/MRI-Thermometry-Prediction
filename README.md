# BrainSeg-Setup
**Segmentation Code:**
  <br />-"segmentation_all.ipynb"
  <br />-The root directory contains all "LP-xxxx" files. This must include anatomicalProbesEye and temperatureData.
  <br />-References/Explanations are included in code at the beginning of each major function

**Resampling/Cropping Code:**
<br />  -"resample_crop.ipynb"
  <br />-Requires MTLE files as root directory
  <br />-remove final "break" in bottom cell to run with all patients
  <br />-References/Explanations are included in code at the beginning of each major function

**Utils Code:**
<br /> -"utils.ipynb"
<br /> -removes t2w MRIs (as well as any other specified ones) using variable called bad_list
<br /> -combines segmentation and anatomical MRIs probesEye
