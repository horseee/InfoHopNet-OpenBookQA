import os
import codecs
import numpy as np
import torch

from modules import get_segment_ids, get_mask_ids

def eval_model(model, data_iter, dataset_len, model_name):
    dev_correct = 0 
    with torch.no_grad():
        for choices, answer in data_iter:
            model.eval()
            if model_name == 'bert':
                input_ids = torch.stack(choices, 1) # batch_size * 4 * seq_len
                segment_ids = get_segment_ids(input_ids)
                mask_ids = get_mask_ids(input_ids)
                output = model.forward(input_ids, segment_ids, mask_ids)
            elif model_name == 'InfoHopNet':
                data = []
                for i in range(4):
                    start_index = 20 * i
                    choice = [choices[start_index], 
                            torch.stack(choices[start_index + 1: start_index + 10], 1), 
                            torch.stack(choices[start_index + 10: start_index + 20], 1)]
                    data.append(choice)
                output = model(data, is_training = True)[0]
            elif model_name == 'MemoryHopNet':
                texts = []
                for i in range(4):
                    texts.append(torch.stack(choices[i*20 + 2: i*20 + 11], 1))
                output = model(choices[0], choices[1:-1:20], texts)
            else:
                data = []
                for i in range(4):
                    data.append((choices[2 * i], choices[2 * i + 1]))
                output = model(data)
                        
            pred = torch.max(output, 1)[1]
            dev_correct += torch.eq(pred, answer).cpu().sum().item()

    dev_acc = np.round(dev_correct/dataset_len, 3)
    return dev_acc
        
