# python main.py --pb_type=maxcut --fr="netket" --input_size 200 1 --model_name="rbm" --optimizer="sr" --batch_size=1024 --learning_rate=0.05 --num_of_iterations=50 --num_trials=1
# python main.py --pb_type=maxcut --fr="netket" --input_size 200 1 --model_name="rbm" --optimizer="sr" --batch_size=2048 --learning_rate=0.05 --num_of_iterations=50 --num_trials=1
# python main.py --pb_type=maxcut --fr="netket" --input_size 200 1 --model_name="rbm" --optimizer="sr" --batch_size=4096 --learning_rate=0.05 --num_of_iterations=50 --num_trials=1
# python main.py --pb_type=maxcut --fr="netket" --input_size 250 1 --model_name="rbm" --optimizer="sr" --batch_size=4096 --learning_rate=0.05 --num_of_iterations=50 --num_trials=1
# python main.py --pb_type=maxcut --fr="manopt" --input_size 50 1 --num_trials=1
# python main.py --pb_type=maxcut --fr="manopt" --input_size 70 1 --num_trials=1
# python main.py --pb_type=maxcut --fr="manopt" --input_size 90 1 --num_trials=1
# python main.py --pb_type=maxcut --fr="manopt" --input_size 100 1 --num_trials=1
# python main.py --pb_type=maxcut --fr="manopt" --input_size 150 1 --num_trials=1
python main.py --pb_type=maxcut --fr="manopt" --input_size 200 1 --num_trials=3
# python main.py --pb_type=maxcut --fr="manopt" --input_size 250 1 --num_trials=1

# python main.py --pb_type=maxcut --fr="debug" --input_size 70 1 --num_trials=1

# python main.py --pb_type=maxcut --fr="sdp_BM" --input_size 200 1 --num_trials=1 --random_seed=310

# --------------------------------------before 04/10--------------------------------------------------
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 50 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_BM" --input_size 50 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_SCS" --input_size 50 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_CVXOPT" --input_size 50 1 --num_trials=10

# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 70 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_BM" --input_size 70 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_SCS" --input_size 70 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_CVXOPT" --input_size 70 1 --num_trials=10

# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 90 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_BM" --input_size 90 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_SCS" --input_size 90 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_CVXOPT" --input_size 90 1 --num_trials=10

# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 100 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_BM" --input_size 100 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_SCS" --input_size 100 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_CVXOPT" --input_size 100 1 --num_trials=10

# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 150 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_BM" --input_size 150 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_SCS" --input_size 150 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_CVXOPT" --input_size 150 1 --num_trials=10

# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 200 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_BM" --input_size 200 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_SCS" --input_size 200 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_CVXOPT" --input_size 200 1 --num_trials=10

# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 250 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_BM" --input_size 250 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_SCS" --input_size 250 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="sdp_CVXOPT" --input_size 250 1 --num_trials=10


# --------------------------------------before 04/10--------------------------------------------------
# # maxcut runs
# python main.py --pb_type=maxcut --fr="random_cut" --input_size 50 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 50 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 50 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 50 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 50 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=10


# python main.py --pb_type=maxcut --fr="random_cut" --input_size 70 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 70 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 70 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 70 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 70 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=10


# python main.py --pb_type=maxcut --fr="random_cut" --input_size 90 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 90 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 90 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 90 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 90 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=10


# python main.py --pb_type=maxcut --fr="random_cut" --input_size 100 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 100 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 100 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 100 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 100 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=10


# python main.py --pb_type=maxcut --fr="random_cut" --input_size 150 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 150 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 150 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 150 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 150 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=10


# python main.py --pb_type=maxcut --fr="random_cut" --input_size 200 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 200 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 200 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 200 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 200 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=4


# python main.py --pb_type=maxcut --fr="random_cut" --input_size 250 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 250 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 250 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 250 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 250 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=10



# python main.py --pb_type=maxcut --fr="netket" --input_size 50 1 --model_name="rbm" --optimizer="adamax" --batch_size=128 --learning_rate=0.005 --num_of_iterations=100 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 70 1 --model_name="rbm" --optimizer="adamax" --batch_size=128 --learning_rate=0.005 --num_of_iterations=100 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 90 1 --model_name="rbm" --optimizer="adamax" --batch_size=128 --learning_rate=0.005 --num_of_iterations=100 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 100 1 --model_name="rbm" --optimizer="adamax" --batch_size=128 --learning_rate=0.005 --num_of_iterations=100 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 150 1 --model_name="rbm" --optimizer="adamax" --batch_size=128 --learning_rate=0.005 --num_of_iterations=100 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 200 1 --model_name="rbm" --optimizer="adamax" --batch_size=128 --learning_rate=0.005 --num_of_iterations=100 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 250 1 --model_name="rbm" --optimizer="adamax" --batch_size=128 --learning_rate=0.005 --num_of_iterations=100 --num_trials=10


# python main.py --pb_type=maxcut --fr="netket" --input_size 50 1 --model_name="rbm" --optimizer="sr" --batch_size=1024 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 70 1 --model_name="rbm" --optimizer="sr" --batch_size=1024 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 90 1 --model_name="rbm" --optimizer="sr" --batch_size=1024 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 100 1 --model_name="rbm" --optimizer="sr" --batch_size=1024 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 150 1 --model_name="rbm" --optimizer="sr" --batch_size=1024 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 200 1 --model_name="rbm" --optimizer="sr" --batch_size=1024 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 250 1 --model_name="rbm" --optimizer="sr" --batch_size=1024 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10


# --------------------------------------before 03/26--------------------------------------------------
# python main.py --pb_type=maxcut --fr="RL" --input_size 50 1 --learning_rate=0.0005 --batch_size=64 --epochs=3000 --depth=2 --num_trials=1 --num_gpu=0
# python main.py --pb_type=maxcut --fr="RL" --input_size 70 1 --learning_rate=0.0005 --batch_size=64 --epochs=3000 --depth=2 --num_trials=1 --num_gpu=0
# python main.py --pb_type=maxcut --fr="RL" --input_size 90 1 --learning_rate=0.0005 --batch_size=64 --epochs=3000 --depth=2 --num_trials=1 --num_gpu=0


# python main.py --pb_type=maxcut --fr="RL" --input_size 50 1 --num_trials=1
# python main.py --pb_type=maxcut --fr="RL" --input_size 70 1 --num_trials=1
# python main.py --pb_type=maxcut --fr="RL" --input_size 90 1 --num_trials=1

# python main.py --pb_type=maxcut --fr="flowket" --input_size 50 1 --model_name="ar1" --optimizer="adam" --batch_size=128 --learning_rate=0.01 --num_of_iterations=100 --num_trials=10


# python main.py --pb_type=maxcut --fr="flowket" --input_size 50 1 --model_name="my_rbm" --optimizer="adam" --batch_size=1024 --learning_rate=0.05 --num_of_iterations=20 --num_trials=10
# python main.py --pb_type=maxcut --fr="flowket" --input_size 70 1 --model_name="my_rbm" --optimizer="adam" --batch_size=1024 --learning_rate=0.05 --num_of_iterations=20 --num_trials=10
# python main.py --pb_type=maxcut --fr="flowket" --input_size 90 1 --model_name="my_rbm" --optimizer="adam" --batch_size=1024 --learning_rate=0.05 --num_of_iterations=20 --num_trials=10

# python main.py --pb_type=maxcut --fr="flowket" --input_size 50 1 --model_name="ar1" --optimizer="adam" --batch_size=128 --learning_rate=0.1 --num_of_iterations=100 --num_trials=1

# --------------------------------------before 03/26--------------------------------------------------
# # maxcut runs
# python main.py --pb_type=maxcut --fr="random_cut" --input_size 50 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 50 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 50 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 50 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 50 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=10


# python main.py --pb_type=maxcut --fr="random_cut" --input_size 70 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 70 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 70 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 70 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 70 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=10


# python main.py --pb_type=maxcut --fr="random_cut" --input_size 90 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 90 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 90 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 90 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 90 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=10


# python main.py --pb_type=maxcut --fr="random_cut" --input_size 100 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 100 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 100 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 100 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 100 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=10


# python main.py --pb_type=maxcut --fr="random_cut" --input_size 150 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 150 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 150 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 150 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 150 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=10


# python main.py --pb_type=maxcut --fr="random_cut" --input_size 200 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 200 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 200 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 200 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 200 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=10

# python main.py --pb_type=maxcut --fr="random_cut" --input_size 300 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 300 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 300 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 300 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 300 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=10

# python main.py --pb_type=maxcut --fr="random_cut" --input_size 500 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="greedy_cut" --input_size 500 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="goemans_williamson" --input_size 500 1 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 500 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50 --num_trials=10
# python main.py --pb_type=maxcut --fr="netket" --input_size 500 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100 --num_trials=10



# ----------------------------------------------------------------------------------------------------------
# # spinglass runs
# python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="mlp1" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="conv_net" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50

# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="mlp1" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="conv_net" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50

# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="mlp1" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="conv_net" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50

# python main.py --pb_type=spinglass --fr=netket --input_size 6 6 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 6 6 --model_name="mlp1" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50
# python main.py --pb_type=spinglass --fr=netket --input_size 6 6 --model_name="conv_net" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=50


# python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="mlp1" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="mlp2" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=spinglass --fr=netket --input_size 3 3 --model_name="conv_net" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100

# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="mlp1" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="mlp2" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --model_name="conv_net" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100

# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="mlp1" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="mlp2" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --model_name="conv_net" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100

# python main.py --pb_type=spinglass --fr=netket --input_size 6 6 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=spinglass --fr=netket --input_size 6 6 --model_name="mlp1" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=spinglass --fr=netket --input_size 6 6 --model_name="mlp2" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=spinglass --fr=netket --input_size 6 6 --model_name="conv_net" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100




# # maxcut runs
# python main.py --pb_type=maxcut --fr=netket --input_size 10 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 11 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 12 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 13 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 14 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 15 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 16 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 17 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 18 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 19 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 20 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 21 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 22 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 23 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 24 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 25 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 26 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 27 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 28 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30
# python main.py --pb_type=maxcut --fr=netket --input_size 29 1 --model_name="rbm" --optimizer="sr" --batch_size=128 --learning_rate=0.05 --num_of_iterations=30

# python main.py --pb_type=maxcut --fr=netket --input_size 10 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 11 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 12 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 13 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 14 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 15 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 16 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 17 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 18 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 19 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 20 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 21 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 22 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 23 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 24 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 25 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 26 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 27 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 28 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 29 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 50 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 60 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 70 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 80 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 90 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 100 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 150 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 200 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100
# python main.py --pb_type=maxcut --fr=netket --input_size 500 1 --model_name="rbm" --optimizer="sgd" --batch_size=128 --learning_rate=0.05 --num_of_iterations=100


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
# python main.py --pb_type=maxcut --fr=flowket --input_size 20 --model_name="ar1" --optimizer="adam" --learning_rate=0.05 --batch_size=128 --log_interval=5 --num_of_iterations=100 --random_seed=117
# python main.py --pb_type=maxcut --fr=flowket --input_size 20 1 --model_name="ar2" --optimizer="adam" --learning_rate=0.001 --batch_size=128 --log_interval=5 --num_of_iterations=100 --random_seed=117
# python main.py --pb_type=maxcut --fr=netket --input_size 20 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40

# python main.py --pb_type=spinglass --fr=flowket --input_size 5 5 --model_name="ar2" --optimizer="adam" --learning_rate=0.001 --batch_size=128 --log_interval=5 --num_of_iterations=100 --random_seed=117
# python main.py --pb_type=spinglass --fr=netket --input_size 5 5 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40


# python main.py --pb_type=maxcut --fr=netket --input_size 12 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=maxcut --fr=netket --input_size 13 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=maxcut --fr=netket --input_size 14 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=maxcut --fr=netket --input_size 15 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=maxcut --fr=netket --input_size 40 1 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=spinglass --fr=netket --input_size 4 4 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
# python main.py --pb_type=spinglass --fr=netket --input_size 7 7 --optimizer="sr" --learning_rate=0.05 --num_of_iterations=40
