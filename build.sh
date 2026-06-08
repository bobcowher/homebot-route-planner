source ~/anaconda3/etc/profile.d/conda.sh

conda env list | grep -q "^car-racing " || conda create -n car-racing python=3.12 -y

conda activate sac-homebot

#pip install -r requirements.txt

python -u ./train.py
