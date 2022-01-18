# DATL
Source Code for the ESANN 2021 Submission 'Domain Adversarial Tangent Learning towards Interpretable Domain Adaptation'

Contact: christoph.raab@fhws.de <br>
If you got problems with this implementation, feel free to write me an email.

## Installation
1. Download the datasets with the links below.

## Training
For a simple training-evaluation demo run with preset parameters, you can use the following commands for training on **Office-31 A->W**<br>

Train network via
```python train_datl.py --source_dir Office-31/images/amazon/ --target_dir Office-31/images/webcam ```
1. The script trains datl on amazon vs webcam given the specified image folder paths
2.   See the Args-Parser parameter description in the file for the documentation of the parameters.
3. The trained models are stored in models

## Explainability

For the explainability results run 



## Datasets
### Office-31
Office-31 dataset can be found [here](https://drive.google.com/file/d/11nywfWdfdBi92Lr3y4ga2Cu4_-FpWKUC/view?usp=sharing).

### Image-clef
Image-Clef dataset can be found [here](https://drive.google.com/file/d/1lu1ouDoeucW8MgmaKVATwNt5JT7Uvk62/view?usp=sharing).
