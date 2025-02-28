git submodule init
git submodule update
cd verl
pip install -e .
cd ..
python scripts/download_dataset.py
pip install -r requirements.txt
# python verl/examples/data_preprocess/gsm8k.py --local_dir data/gsm8k