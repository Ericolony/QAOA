
# spinglass runs
python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="mlp1" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="conv_net" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50

python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="mlp1" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="conv_net" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50

python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="mlp1" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="conv_net" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50

python main.py --pb_type=spinglass --fr=netket --input_size 6 6 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
python main.py --pb_type=spinglass --fr=netket --input_size 6 6 --model_name="mlp1" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
python main.py --pb_type=spinglass --fr=netket --input_size 6 6 --model_name="conv_net" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50


python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="mlp1" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="mlp2" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="conv_net" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200

python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="mlp1" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="mlp2" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="conv_net" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200

python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="mlp1" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="mlp2" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="conv_net" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200

python main.py --pb_type=spinglass --fr=netket --input_size 6 6 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=spinglass --fr=netket --input_size 6 6 --model_name="mlp1" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=spinglass --fr=netket --input_size 6 6 --model_name="mlp2" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=spinglass --fr=netket --input_size 6 6 --model_name="conv_net" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200




# maxcut runs
python main.py --pb_type=maxcut --fr=netket --input_size 10 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 11 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 12 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 13 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 14 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 15 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 16 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 17 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 18 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 19 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 20 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 21 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 22 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 23 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 24 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 25 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 26 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 27 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 28 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
python main.py --pb_type=maxcut --fr=netket --input_size 29 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30

python main.py --pb_type=maxcut --fr=netket --input_size 10 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 11 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 12 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 13 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 14 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 15 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 16 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 17 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 18 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 19 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 20 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 21 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 22 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 23 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 24 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 25 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 26 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 27 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 28 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 29 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 50 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 60 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 70 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 80 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 90 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 100 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 150 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 200 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200
python main.py --pb_type=maxcut --fr=netket --input_size 500 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=200


#######################################################################################################################################################

# python main.py --pb_type=spinglass --fr=netket --input_size 2 2 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 2 2 --model_name="mlp1" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 2 2 --model_name="mlp2" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50

# python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="mlp1" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="mlp2" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50

# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="mlp1" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="mlp2" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50

# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="mlp1" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="mlp2" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50


# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="conv_net" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="conv_net" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50


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
