<<<<<<< HEAD
# python main.py --pb_type=maxcut --fr=flowket --input_size 20 --model_name="ar1" --optimizer="adam" --learning_rate=0.05 --batch_size=128 --log_interval=5 --num_of_iterations=200 --random_seed=117
# python main.py --pb_type=maxcut --fr=flowket --input_size 20 1 --model_name="ar2" --optimizer="adam" --learning_rate=0.001 --batch_size=128 --log_interval=5 --num_of_iterations=200 --random_seed=117
python main.py --pb_type=maxcut --fr=netket --input_size 20 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40

# python main.py --pb_type=spinglass --fr=flowket --input_size 5 5 --model_name="ar2" --optimizer="adam" --learning_rate=0.001 --batch_size=128 --log_interval=5 --num_of_iterations=200 --random_seed=117
# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40

=======
# python try.py netket -input_size=20 -num_of_iterations=200
# python try.py flowket -input_size=20 -model_name='ar' -l=0.001 -batch_size=128 -num_of_iterations=500 -random_seed=110
# python try.py netket -input_size=20 -num_of_iterations=20
# python benchmark.py flowket
# python benchmark.py netket
# python cqo.py
# python auto_regressive.py

python main.py --fr=flowket --input_size=20 --model_name='ar' --learning_rate=0.001 --batch_size=128 --log_interval=20 --num_of_iterations=500 --random_seed=110
python main.py --fr=netket --input_size=20 --num_of_iterations=200
>>>>>>> b1250baaa20a9bb578d8d052b6ec67bd5aa80232
