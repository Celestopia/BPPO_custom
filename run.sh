python main.py --dataset_name "ip" \
    --result_dir "logs" \
    --train_data_path "datasets\\inverted_pendulum_99900_127_2_1.pkl" \
    --test_data_path "datasets\\inverted_pendulum_100_127_2_1.pkl" \
    --action_reward_scale 10 \

python main.py --dataset_name "power" \
    --result_dir "logs" \
    --train_data_path "datasets\\power_99900_31_18_9.pkl" \
    --test_data_path "datasets\\power_100_31_18_9.pkl" \
    --action_reward_scale 1000 \

python main.py --dataset_name "kuramoto" \
    --result_dir "logs" \
    --train_data_path "datasets\\kuramoto_99900_15_8_8.pkl" \
    --test_data_path "datasets\\kuramoto_100_15_8_8.pkl" \
    --action_reward_scale 0.001 \

python main.py --dataset_name "burgers" \
    --result_dir "logs" \
    --train_data_path "datasets\\burgers_90000_10_128_128.pkl" \
    --test_data_path "datasets\\burgers_50_10_128_128.pkl" \
    --action_reward_scale 0.01 \