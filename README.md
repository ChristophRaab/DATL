# DATL
Source Code for the ESANN 2021 Submission 'Domain Adversarial Tangent Learning towards Interpretable Domain Adaptation'

Contact: christoph.raab@fhws.de <br>
If you got problems with this implementation, feel free to write me an email.

## Installation
1. We provide the conda requirements in [conda_requirements.txt](https://github.com/ChristophRaab/DATL/blob/main/conda_requirements.txt) or pip requirements in [pip_requirements.txt](https://github.com/ChristophRaab/DATL/blob/main/pip_requirements.txt). After installation you should be able to run the code in this repository.

2. First, download the dataset your are prefering. The link can be found below.
3. After download, unzip the files and place the output in the desired directory.

## Demo
For a simple training-evaluation demo run with preset parameters, you can use the following commands for training on

**Office-31 A->W**<br>
`train_datl.py --source_dir Office-31/images/amazon/ --target_dir Office-31/images/webcam --subspace_dim 128`<br>
The script trains datl on amazon vs webcam given the specified image folder paths

## Training and Inference
1. The network can be trained via train_datl.py on any Office-31 combination
   See the Args-Parser parameter description in the file for the documentation of the parameters.

2. The trained models are stored in models


### Office-31 Dataset
Office-31 dataset can be found [here](https://drive.google.com/file/d/11nywfWdfdBi92Lr3y4ga2Cu4_-FpWKUC/view?usp=sharing).
