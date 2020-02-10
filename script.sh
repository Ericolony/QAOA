# python try.py netket -input_size=20 -num_of_iterations=200
# python try.py flowket -input_size=20 -model_name='ar' -l=0.001 -batch_size=128 -num_of_iterations=500 -random_seed=110
# python try.py netket -input_size=20 -num_of_iterations=20
# python benchmark.py flowket
# python benchmark.py netket
# python cqo.py
# python auto_regressive.py

python main.py --fr=flowket --input_size=20 --model_name='ar' --learning_rate=0.001 --batch_size=128 --log_interval=20 --num_of_iterations=500 --random_seed=110
python main.py --fr=netket --input_size=20 --num_of_iterations=200