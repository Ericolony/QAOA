python main.py --pb_type=maxcut --fr="random_cut" --input_size 20 1 --num_trials=1 --random_seed=666

python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 20 1 --num_trials=1 --random_seed=666

python main.py --pb_type=maxcut --fr="manopt" --input_size 20 1 --num_trials=1 --random_seed=666

python main.py --pb_type=maxcut --fr="netket" --input_size 20 1 --num_trials=1 --model_name="rbm" --optimizer="sgd" --batch_size=1024 --learning_rate=0.05 --num_of_iterations=50 --random_seed=666
