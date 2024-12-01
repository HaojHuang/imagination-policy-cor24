import pickle


path = '/home/hhj/Documents/fourier_transporter/RLBench/rlbench_data/stack_cups/variation2/variation_descriptions.pkl'
with open(path, 'rb') as f:
    des = pickle.load(f)
    print(des)
    print(len(des))
