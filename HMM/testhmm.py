# -*- coding:utf-8 -*-
# Filename: test_weather.py
# Author：hankcs
# Date: 2016-08-06 PM6:04
import numpy as np

import hmm
'''
states = ('Healthy', 'Fever')

observations = ('normal', 'cold', 'dizzy')

start_probability = {'Healthy': 0.6, 'Fever': 0.4}

transition_probability = {
    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
    'Fever': {'Healthy': 0.4, 'Fever': 0.6},
}

emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}
'''
states = ('sunny', 'cloudy', 'rainy')
observations = ('dry','dryish','damp','soggy')

start_probability ={'sunny':0.6,'cloudy':0.2,'rainy':0.2}

transition_probability = {
    'sunny':{'sunny':0.5,'cloudy':0.375,'rainy':0.125},
    'cloudy':{'sunny':0.25,'cloudy':0.125,'rainy':0.625},
    'rainy':{'sunny':0.25,'cloudy':0.375,'rainy':0.375}
}

emission_probability ={
    'sunny':{'dry':0.6,'dryish':0.2,'damp':0.15,'soggy':0.05},
    'cloudy':{'dry':0.25,'dryish':0.25,'damp':0.25,'soggy':0.25},
    'rainy':{'dry':0.05,'dryish':0.1,'damp':0.35,'soggy':0.50}
}

def generate_index_map(lables):
    index_label = {}
    label_index = {}
    i = 0
    for l in lables:
        index_label[i] = l
        label_index[l] = i
        i += 1
    return label_index, index_label


states_label_index, states_index_label = generate_index_map(states)
observations_label_index, observations_index_label = generate_index_map(observations)


def convert_observations_to_index(observations, label_index):
    list = []
    for o in observations:
        list.append(label_index[o])
    return list


def convert_map_to_vector(map, label_index):
    v = np.empty(len(map), dtype=float)
    for e in map:
        v[label_index[e]] = map[e]
    return v


def convert_map_to_matrix(map, label_index1, label_index2):
    m = np.empty((len(label_index1), len(label_index2)), dtype=float)
    for line in map:
        for col in map[line]:
            m[label_index1[line]][label_index2[col]] = map[line][col]
    return m


A = convert_map_to_matrix(transition_probability,states_label_index,states_label_index)
print (A)
B = convert_map_to_matrix(emission_probability,states_label_index,observations_label_index)
print (B)
observations_index = convert_observations_to_index(observations,observations_label_index)
print (observations_index)
pi = convert_map_to_vector(start_probability,states_label_index)
print (pi)

h = hmm.HMM(A, B, pi)
V, p = h.viterbi(observations_index)
print(" " * 7, " ".join(("%10s" % observations_index_label[i]) for i in observations_index))
for s in range(0, 2):
    print("%7s: " % states_index_label[s] + " ".join("%10s" % ("%f" % v) for v in V[s]))
print('\nThe most possible states and probability are:')
p, ss = h.state_path(observations_index)
for s in ss:
    print(states_index_label[s],)
print(p)

# run a baum_welch_train
observations_data, states_data = h.simulate(10)
print(observations_data)
print(states_data)
'''
guess = hmm.HMM(np.array([[0.5, 0.5],
                          [0.5, 0.5]]),#A
                np.array([[0.3, 0.3, 0.3],
                          [0.3, 0.3, 0.3]]),#B
                np.array([0.5, 0.5])#π
                )

F = guess._forward(observations_data)
print("forward:")
print(F)
guess.baum_welch_train(observations_data)
states_out = guess.state_path(observations_data)[1]
p = 0.0
for s in states_data:
    if next(states_out) == s: p += 1

print(p / len(states_data))
'''
h = hmm.HMM(A,B,pi)
# 人为定义的海藻状态序列
obs_seq = ('dry','damp','soggy')
obs_seq_index = convert_observations_to_index(obs_seq,observations_label_index)
# 计算P(o|lambda)
F = h._forward(obs_seq_index)
print ("forward: P(O|lambda) = %f" %sum(F[:,-1]))
X = h._backward(obs_seq_index)
print ("backward: P(O|lambda) = %f" %sum(X[:,0]*pi*B[:,0]))

# 计算P(I|o)
p,ss = h.state_path(obs_seq_index)
path = []
for s in ss:
    path.append(states_index_label[s])

print("最有可能的隐藏序列为：" ,path)
print("viterbi: P(I|O) =%f"% p)
