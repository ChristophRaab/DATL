# DATL
Source Code for the ESANN 2021 Paper Submission 'Domain Adversarial Tangent Learning towards Interpretable Domain Adaptation'

Contact: christoph.raab@fhws.de <br>
If you got problems with this implementation, feel free to write me an email.

## Installation
1. We provide the requirements in [requirements.txt](https://github.com/sub-dawg/ASAN/blob/master/pytorch.yml). After installation you should be able to run the code in this repository.

2. First, download the dataset your are prefering. The link can be found below.
3. After download, unzip the files and place the output folders as they are in the directory [images](https://github.com/sub-dawg/ASAN/blob/master/images/).

## Demo
For a simple training-evaluation demo run with preset parameters, you can use the following commands for training on

**Office-31 A->W**<br>
`train_datn.py --tl RSL --s_dset_path data/office/amazon.txt --t_dset_path data/office/webcam.txt --test_interval 100 --num_workers 12 --sn True --k 11 --tllr 0.001`

## Training and Inference
1. The network can be trained via train_datl.py
   See the Args-Parser parameter description in the file for the documentation of the parameters.

2. The trained models are obtainable under models


### Office-31 Dataset
Office-31 dataset can be found [here](https://drive.google.com/file/d/11nywfWdfdBi92Lr3y4ga2Cu4_-FpWKUC/view?usp=sharing).
