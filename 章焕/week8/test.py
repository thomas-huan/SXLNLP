import json
import random

import torch
import torch.nn as nn

# tensor1 = torch.tensor([2.0,4.0])
# tensor2 = torch.tensor([4.0,2.0])
#
# t1 = torch.nn.functional.normalize(tensor1, dim=-1) #模长
# t2 = torch.nn.functional.normalize(tensor2, dim=-1) #模长
# mul = torch.mul(tensor1, tensor2)
# cosine = torch.sum(torch.mul(tensor1, tensor2)) #外积再求个  这不就等于内积吗
#
#
# print(t1)
# print(t2)
# print(mul)
# print(cosine)
mydict = {}
with open("../data/train.json", encoding="utf8") as f:
    for line in f:
        line = json.loads(line)
        if isinstance(line, dict):
            questions = line["questions"]
            label = line["target"]
            mydict[label] = questions
print(mydict)



with open("../data/triplet_train.json", "a", encoding="utf8") as f:

    new_dict={}
    for _ in range(1000):
        data1 = random.choice(list(mydict.items()))
        data2 = random.choice(list(mydict.items()))
        if data1[0] == data2[0]:
            continue

        #print(data1[0]+" "+random.choice(data1[1])+" "+random.choice(data2[1]))
        new_dict["anchor"] = data1[0]
        new_dict["positive"] = random.choice(data1[1])
        new_dict["negative"] = random.choice(data2[1])
        f.write(json.dumps(new_dict,ensure_ascii=False)+"\n")