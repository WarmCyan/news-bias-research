#!/bin/python3

import json

problem = "biased"
embedding = "w2v"
embedding_shape = "avg_std"
model_type = "nn"
arch_num = 1
layers = [128, 1] # will have to read json string
model_num = 0
tag = ""

problem_input = input("Problem: (b)iased, bias_(d)irection, (r)eliability [biased]> ")
if problem_input == "b":
    problem = "biased"
elif problem_input == "d":
    problem = "bias_direction"
elif problem_input == "r":
    problem = "reliability"
print(problem)

embedding_input = input("Embedding: [w2v]> ")
if embedding_input != "":
    embedding = embedding_input
print(embedding)

embedding_shape_input = input("Shape: (a)vg, avg_st(d), (s)equence [avg_std]> ")
if embedding_shape_input == "a":
    embedding_shape = "avg"
elif embedding_shape_input == "d":
    embedding_shape = "avg_std"
elif embedding_shape_input == "s":
    embedding_shape = "sequence"
print(embedding_shape)

model_type_input = input("Model type: (n)n, (l)stm, (s)vm [nn]> ")
if model_type_input == "n":
    model_type = "nn"
elif model_type_input == "l":
    model_type = "lstm"
elif model_type_input == "s":
    model_type = "svm"
print(model_type)

arch_num_input = input("Arch number: [1]> ")
if arch_num_input != "":
    arch_num = int(arch_num_input)
print(arch_num)

layers_input = input("Layers: [[128, 1]]> ")
if layers_input != "":
    layers = json.loads(layers_input)
print(layers)

model_num_input = input("Starting model number:> ")
model_num = int(model_num_input)
print(model_num)

tag_input = input("Experiment tag:> ")
tag = tag_input
print(tag)

experiments = []
for i in range(0, 10):
    experiment_model_num = model_num + i 
    experiment = {
            "type": "model",
            "selection_problem": problem,
            "selection_test_fold": i,
            "selection_source": "", 
            "selection_test_source": "",
            "selection_count": 1000,
            "selection_random_seed": 13,
            "selection_reject_minimum": 300,
            "selection_overwrite": False,
            "embedding_type": embedding,
            "embedding_shape": embedding_shape,
            "embedding_overwrite": False,
            "verbose": False,
            "model_type": model_type,
            "model_arch_num": arch_num,
            "model_layer_sizes": layers,
            "model_maxlen": 500,
            "model_batch_size": 128,
            "model_learning_rate": 0.001,
            "model_epochs": 200,
            "model_num": experiment_model_num,
            "experiment_tag": tag
            }
    experiments.append(experiment)

with open("../experiments/" + tag + ".json", 'w') as outfile:
    json.dump(experiments, outfile, indent=4)
print("Wrote out '../experiments/" + tag + ".json'")
