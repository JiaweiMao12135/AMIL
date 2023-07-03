# IKGM: A novel method for identifying key genes in macroev-olution based on deep learning with attention mechanism

## Python implementation
### Environment and Packages

By default your device has conda installed


Creating a virtual environment and activating it:

``conda create -n your_name python=3.8``

``conda activate your_name``


Packages list is in file: condalist.txt, Install all the packages you need to use:

`conda install --yes --file condalist.txt`

### Prepare the genomic (protein) files to be used
#### Example file data in our article
The protein sequences of 34 Lepidoptera were stored in 'data/D/' and 'data/N/' based on the diurnal activity information of the corresponding species, respectively. In addition, the Pfam annotation result files for each species are stored in 'data/lep/'.
### Data pre-processing
You can call datapretreatment.py to perform data preprocessing operations.

In the main function of the datapretreatment.py file, you can set parameters such as the original file path, the generated file path, the k value, etc.

After that, just execute the following command:

`python datapretreatment.py`

After that, two datasets will be generated and divided into training and test sets according to the scale.

### Create input files AND Word2Vec

You can complete the final formatting and word embedding of the input file by using the following commands.


`python create_input_files.py`
### Model 
The main architecture of the hierarchical attention network is stored in model.py

### Training
`python train.py`

### Testing
`python eval.py`

### Obtain attention scores (weights).
When performing the data preprocessing step, two doc.csv files are generated, corresponding to v=1 and v=2 cases. You can use classifiy.py to perform the classification operation on the second column of the doc.csv file and get the attention score.





# hierarchical.pt
The hyper parameters of CrepHAN

# total_10MerVector.txt
download from https://pan.baidu.com/s/1z25RccnIwWUU1crK_JmkZw (code: 0k65)
The dictionary composed of word vectors