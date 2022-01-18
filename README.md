# DATSA - Adversarial Domain Adaptation Network
Source Code and for the Neurocomputing ESANN 2021 Special Issue 'Domain Adversarial Tangent Subspace Alignment for Explainable Domain Adaptation'

Contact: christophraab@outlook.de <br>
If you got problems with this implementation, feel free to write me an email.

## Installation
1. Download the datasets with the links below.
2. Run ```pip install -e .[all]``` to install the **DATL** package.

## Training
For a simple training-evaluation demo run with preset parameters, you can use the following commands for training on **Office-31 A->W**<br>

Train network via
```cd datl && python train_datl.py --source_dir Office-31/images/amazon/ --target_dir Office-31/images/webcam ```<br>
*Note that source_dir and target_dir must be replaced with your dataset locations.*<br>
1. The script trains datl on amazon vs webcam given the specified image folder paths.
2. See the Args-Parser parameter description in the file for the documentation of the parameters.
3. The best model is stored in ```models/```.

## Explainability

For the explainability results on **Office-31 A->W** do: <br>  

1. Replace dataset paths in line 11 and 12 in ```study.py```. 
2. ```cd datl && python explainability.py --source amazon --target webcam```
3. The Siamese Translations (STs) are stored in ```results/´´´.
4. T-Sne plots of STs and features are stored in ```plots/``´´.

## Reproduce Performance results
For the explainability results on **Office-31** do: <br>  
1. Replace dataset paths in line 11 and 12 in ```study.py```. 
2. ```cd datl && python study.py --dset office```<br>
3. Results are stored as csv file ```results/```

## Datasets
### Office-31
Office-31 dataset can be found [here](https://drive.google.com/file/d/11nywfWdfdBi92Lr3y4ga2Cu4_-FpWKUC/view?usp=sharing).

### Image-clef
Image-Clef dataset can be found [here](https://drive.google.com/file/d/1lu1ouDoeucW8MgmaKVATwNt5JT7Uvk62/view?usp=sharing).
