@echo off
python src/data/synth_generator.py --out data/synthetic --n_patients 300 --n_taxa 100 --ae_rate 0.12
python src/train.py --data_dir data/synthetic --out outputs/xgb_model.pkl
python src/evaluate.py --data_dir data/synthetic --model outputs/xgb_model.pkl

