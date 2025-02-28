git submodule init
git submodule update
pip install -r requirements.txt
cd verl
pip install -e .
cd ..
python scripts/download_dataset.py
# python verl/examples/data_preprocess/gsm8k.py --local_dir data/gsm8k