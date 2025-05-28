*An NN-improved SST turbulence model with varying Prt (turbulence heat flux term) and β (Reynolds stress term) was trained, and the NN-SST turbulence model is coupled with the HISA solver through the TensorFlow API*

**Environment configuration:**

[1] The EnKF algorithm adopts the open-source data assimilation and field inversion framework *DAFI*, so the inversion environment needs to be configured first by following the instructions at:
[https://dafi.readthedocs.io/en/latest/install.html#](https://dafi.readthedocs.io/en/latest/install.html#)

[2] This work uses TensorFlow 2.13.0, and the corresponding environment should be set up using pip by installing the appropriate version and downloading the matching API libraries at: [https://www.tensorflow.org/install/lang_c#](https://www.tensorflow.org/install/lang_c#).

[3] The HISA solver is a C++ based tool for computing compressible transonic and supersonic flow (https://hisa.gitlab.io/#). In this work, this solver is complined based on *OpenFoam v2012* coupled with ANN enhanced turbulence heat flux and Reynolds stress.

[4] Compile the feature extraction program *writeFieldsMLr4.C* by running *wmake*, which is used to extract NN inputs during the training process.

**Models training:**

Once the above environment is installed, the training program can be executed by running: *python /path/to/bin/dafi dafi.in*, by running the *pltmisfit.py* to plot the convergence history
![4](https://github.com/user-attachments/assets/4b22863d-aa48-484c-91a7-9f45cfab5ddb)

**Models testing:**

Enable fully coupled model evaluation by setting `betannmodel 1; Prtnnmodel 1;` in the `constant/turbulenceProperties` file, then run `./runsim` in the case directory.



