# python main.py --pb_type=maxcut --fr=flowket --input_size 20 --model_name="ar1" --optimizer="adam" --learning_rate=0.05 --batch_size=128 --log_interval=5 --num_of_iterations=200 --random_seed=117
# python main.py --pb_type=maxcut --fr=flowket --input_size 20 1 --model_name="ar2" --optimizer="adam" --learning_rate=0.001 --batch_size=128 --log_interval=5 --num_of_iterations=200 --random_seed=117
python main.py --pb_type=maxcut --fr=netket --input_size 20 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40

# python main.py --pb_type=spinglass --fr=flowket --input_size 5 5 --model_name="ar2" --optimizer="adam" --learning_rate=0.001 --batch_size=128 --log_interval=5 --num_of_iterations=200 --random_seed=117
# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40

