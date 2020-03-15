python try.py --pb_type=maxcut --fr=netket --input_size 12 1
python try.py --pb_type=maxcut --fr=netket --input_size 15 1
python try.py --pb_type=maxcut --fr=netket --input_size 18 1
python try.py --pb_type=maxcut --fr=netket --input_size 21 1
python try.py --pb_type=maxcut --fr=netket --input_size 24 1


# -------------------------------------------------------------------------------------------------------------
# python main.py --pb_type=maxcut --fr=netket --input_size 25 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40

# python main.py --pb_type=spinglass --fr=netket --input_size 2 2 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40

# -------------------------------------------------------------------------------------------------------------
# python main.py --pb_type=maxcut --fr=flowket --input_size 20 --model_name="ar1" --optimizer="adam" --learning_rate=0.05 --batch_size=128 --log_interval=5 --num_of_iterations=200 --random_seed=117
# python main.py --pb_type=maxcut --fr=flowket --input_size 20 1 --model_name="ar2" --optimizer="adam" --learning_rate=0.001 --batch_size=128 --log_interval=5 --num_of_iterations=200 --random_seed=117
# python main.py --pb_type=maxcut --fr=netket --input_size 20 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40

# python main.py --pb_type=spinglass --fr=flowket --input_size 5 5 --model_name="ar2" --optimizer="adam" --learning_rate=0.001 --batch_size=128 --log_interval=5 --num_of_iterations=200 --random_seed=117
# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40


# python main.py --pb_type=maxcut --fr=netket --input_size 12 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=maxcut --fr=netket --input_size 13 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=maxcut --fr=netket --input_size 14 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=maxcut --fr=netket --input_size 15 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=maxcut --fr=netket --input_size 40 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=spinglass --fr=netket --input_size 7 7 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40