Initial Branch

1.- Install virtualenv

pip install virtualenv


2.- Creat virtual env with enviroments

virtualenv -p python3 venv

3.- Activate virtualenv

source venv/bin/activate

4.- Install requirements

pip install -r requirements.txt

5.- Check python version >3.5
python --version

6.- Prepare intel image classification dataset

unzip downloads/intel-image-classification.zip -d input/intel-image-classification
sudo unzip input/intel-image-classification/seg_train.zip -d input/intel-image-classification/seg_train/

7.- Prepare carvana image masking dataset
unzip downloads/carvana-image-masking-challenge.zip -d input/carvana-image-masking-challenge
sudo unzip input/carvana-image-masking-challenge/train.zip -d input/carvana-image-masking-challenge/
sudo unzip input/carvana-image-masking-challenge/train_masks.zip -d input/carvana-image-masking-challenge/

8.- Prepare standford cars dataset
unzip downloads/stanford-car-dataset-by-classes-folder.zip -d input/stanford-car-dataset-by-classes-folder
sudo unzip input/stanford-car-dataset-by-classes-folder/car_data.zip -d input/stanford-car-dataset-by-classes-folder/car_data


9.- Train first model
cd input_model/
python first_model.py

10.- Train second model
cd inpurt_model/
python second_model.py



#################################3

Test model with frontend




 

