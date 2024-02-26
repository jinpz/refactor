from model_names import model_names_dict
import argparse
from data import get_data
import pickle
import numpy as np
from theorem_expansion import *
from collections import Counter
import pandas as pd
import json
import torch
import copy
from collections import Counter
from shutil import copyfile


def get_proof_level_acc(node_correctness, batch_batch):
    proof_level_acc = torch.zeros((batch_batch[-1].item() + 1,)).to(batch_batch.device)
    for i in range(batch_batch[-1].item() + 1):
        current_correctness = node_correctness[batch_batch == i]
        count = current_correctness.long().sum().item()
        if count != current_correctness.shape[0]:
            proof_level_acc[i] = 0
        else:
            proof_level_acc[i] = 1
    return proof_level_acc


def analyze_node_level_accuracy(y_hat, y, batch_batch):
    y_hat_hard = y_hat.round()
    node_correctness = (y == y_hat.round())
    for i in range(batch_batch[-1].item() + 1):
        current_correctness = node_correctness[batch_batch == i]
        y_hat_hard_current = y_hat_hard[batch_batch == i]
        y_current = y[batch_batch == i]
        print('node accuracy: {0}'.format(current_correctness.float().mean()))
        print('percentage of predicted red nodes over ground truth: {0}'.format(y_hat_hard_current.sum() / y_current.sum()))


def evaluate_loader(loader, model, gpu):
    predictions = []
    labels = []
    for batch in loader:
        if batch.num_graphs == 1:
            if len(batch.y) > args.unexpanded_node_limit_proof:
                continue
        if gpu:
            batch = batch.to(torch.device('cuda'))
        y_hat = model(batch)
        if len(y_hat.shape) == 0:
            predictions.append(float(y_hat))
            labels.append(float(y_hat))
        else:
            predictions.extend(y_hat.tolist())
            labels.extend(batch.y.tolist())
    return predictions, labels


def get_correct_total_stat(correct_proof_names, dict_by_expanding_theorem):
    d = {}
    for correct_proof_name in correct_proof_names:
        name = correct_proof_name[:correct_proof_name.find('variant') - 1]
        expanding_theorem = name[name.find('expand_') + 7:name.find('_in_')]
        if expanding_theorem not in d:
            d[expanding_theorem] = [0, 0]  # correct, total
        d[expanding_theorem][0] += 1
    for theorem in dict_by_expanding_theorem.keys():
        if theorem not in d.keys():
            d[theorem] = [0, 0]
        d[theorem][1] += len(dict_by_expanding_theorem[theorem])
    return d


def change_proof_name(proofs, suffix):
    for k in proofs.keys():
        if 'expand' in k:
            v_list = proofs[k]
            for i in range(len(v_list)):
                v_list[i].name += '_variant_{0}'.format(i) + suffix
        else:
            proofs[k].name += suffix


def check_proof_correct(predictions, labels):
    assert len(predictions) == len(labels)
    predictions = np.round(np.array(predictions))
    num_node_correct = np.sum(predictions == labels)
    if num_node_correct != len(predictions):
        return 0
    else:
        return 1


def check_proof_is_tree(proof_raw, predictions):
    # must have more than one colored node
    adjacency_dict = {}
    new_source = proof_raw[2]
    new_target = proof_raw[1]
    nodes = proof_raw[3]
    assert len(nodes) == len(predictions)
    # check if the proof is a tree
    for i in range(len(predictions)):
        if round(predictions[i]) == 1:
            adjacency_dict[i] = []
    # check if it only has one node
    if len(adjacency_dict) <= 1:
        return 0
    for i in range(len(new_source)):
        if new_source[i] in adjacency_dict.keys() and new_target[i] in adjacency_dict.keys():
            adjacency_dict[new_source[i]].append(new_target[i])
    all_nodes = list(adjacency_dict.keys())
    nodes_with_incoming_edges = []
    for k, v in adjacency_dict.items():
        nodes_with_incoming_edges.extend(v)
    assert len(set(nodes_with_incoming_edges)) == len(nodes_with_incoming_edges)
    if len(all_nodes) - len(nodes_with_incoming_edges) != 1:
        return 0
    else:
        return 1


def find_root_node_proof_tree(proof):
    if round(proof.subst) == 1:
        return proof
    for child in proof.mand_vars:
        res = find_root_node_proof_tree(child)
        if res is not None:
            return res
    for child in proof.hps:
        res = find_root_node_proof_tree(child)
        if res is not None:
            return res
    return None


def proof_has_sub(proof):
    summary = proof.summarize_proof()
    for e in summary:
        if 'sub' in e:
            return True
    return False


def check_proof_meaningful(mm, proof, extracted_proof_name):
    # do this only if proof is already a tree
    root_node = find_root_node_proof_tree(proof)

    # another dfs here, for each red node, its children must have the same color
    a = [root_node]
    while len(a) > 0:
        node = a.pop(0)
        temp = []
        flag = None
        for child in node.mand_vars:
            if flag is None:
                flag = round(child.subst)
            else:
                if round(child.subst) != flag:
                    return None
            temp.append(child)
        for child in node.hps:
            if flag is None:
                flag = round(child.subst)
            else:
                if round(child.subst) != flag:
                    return None
            temp.append(child)
        a = temp + a
    # now we can extract it safely

    print('checking raw {0}'.format(extracted_proof_name))
    raw_verified, _ = mm.verify_custom(proof.expr, proof.summarize_proof(), '', mode='other')
    if not raw_verified:
        # raise NotImplementedError('Serious error about dataset')
        if proof_has_sub(proof):
            print('assumed verified')
        else:
            raise NotImplementedError('Serious error about dataset')
    extracted_proof = extract_potential_meaningful_proof(proof)
    special_flag = classify_special_type(extracted_proof)
    #
    # provide a name
    extracted_proof.name = extracted_proof_name
    print('checking extracted {0}'.format(extracted_proof_name))
    extracted_verified, _ = mm.verify_custom(extracted_proof.expr, extracted_proof.summarize_proof(), '', mode='other')

    if special_flag and extracted_verified:
        raise NotImplementedError('Serious error about dataset')
    if not special_flag and not extracted_verified and raw_verified:
        raise NotImplementedError('Serious error about dataset')

    if not extracted_verified:
        if special_flag:
            print('extracted verification failed expected')

    standardized_extracted_proof = standardize(mm, copy.deepcopy(extracted_proof), change_type=False)

    return standardized_extracted_proof


def classify_special_type(proof):
    special_flag = 0
    # if we had to change the type before standardization
    a = [proof]
    while len(a) > 0:
        node = a.pop(0)
        temp = []
        for child in node.mand_vars:
            if child.type == 'special':
                child.type = '$f'
                special_flag = 1
            temp.append(child)
        for child in node.hps:
            if child.type == 'special':
                child.type = '$e'
                special_flag = 1
            temp.append(child)
        a = temp + a
    return special_flag


def extract_potential_meaningful_proof(proof):
    # do this only the proof is a tree, and for each red node, all its children have the same color
    root_node = find_root_node_proof_tree(proof)
    new_root_node = copy.deepcopy(root_node)

    # another dfs here
    a = [new_root_node]
    while len(a) > 0:
        node = a.pop(0)
        temp = []
        for child in node.mand_vars:
            if round(child.subst) == 1:
                temp.append(child)
            else:
                # this node has children, need to change its type to a special one and replace
                node.type = 'special'
                node.mand_vars = []
                break
        for child in node.hps:
            if round(child.subst) == 1:
                temp.append(child)
            else:
                # should already be replaced special
                assert node.type == 'special'
                node.hps = []
                break
        a = temp + a

    return new_root_node


def standardize(mm, extracted_proof, change_type):
    leaves = extracted_proof.get_leaves(change_type=change_type)
    replace_dict = {}
    used_mand_vars = []
    hps_counter = 0
    labels = mm.labels
    in_scope_labels = mm.in_scope_labels
    used_expr = []  # for avoiding cases wceq.cA wcel.cA
    for i in range(len(leaves)):
        leaf = leaves[i]
        if leaf.type == '$e':
            hps_counter += 1
            leaf.label = extracted_proof.name + '.{0}'.format(hps_counter)
            labels[leaf.label] = ('$e', leaf.expr)  # just a placeholder for the expr, don't use copy here since it will destroy the automatic substitution in propagate
        elif leaf.type == '$f':
            # assert len(leaf.expr) == 2  # not true
            if tuple(leaf.expr) not in replace_dict:
                for k, v in in_scope_labels.items():  # replace only with main scope variables
                    if v[0] == '$f' and v[1][0] == leaf.expr[0] and k not in used_mand_vars and k not in ['sub0', 'sub1', 'sub2']:
                        assert len(v[1]) == 2
                        # avoid accidentally make theorem constrained
                        if tuple(v[1]) in used_expr:
                            continue
                        replace_dict[tuple(leaf.expr)] = k
                        used_mand_vars.append(k)
                        used_expr.append(tuple(v[1]))
                        break
                if tuple(leaf.expr) not in replace_dict:
                    # used up our alphabet
                    print('used up our alphabet')
                    return None
            leaf.label = replace_dict[tuple(leaf.expr)]
            leaf.expr = copy.deepcopy(labels[replace_dict[tuple(leaf.expr)]][1])
            leaf.data = leaf.expr
        else:
            pass
    proof_list = extracted_proof.summarize_proof()
    standardized_extracted_proof = mm.propagate_and_substitute_leaf_hps(proof_list, extracted_proof.name)
    success, _ = mm.verify_custom(standardized_extracted_proof.expr, standardized_extracted_proof.summarize_proof(), '', mode='other')
    if success:
        print('verified {0}'.format(extracted_proof.name))
        return standardized_extracted_proof
    else:
        print('still cannot verify {0}'.format(extracted_proof.name))
        return None


def color_proof_tree(proof, predictions):
    # custom dfs
    visited = []
    a = [proof]
    while len(a) > 0:
        node = a.pop(0)
        visited.append(node)
        temp = []
        for child in node.mand_vars:
            temp.append(child)
        for child in node.hps:
            temp.append(child)
        a = temp + a

    assert len(visited) == len(predictions)
    for i in range(len(visited)):
        visited[i].subst = predictions[i]


def count_proof_name_frequency(mm):
    proofs = mm.proofs
    labels = mm.labels
    proof_count_list = []
    for k, v in proofs.items():
        if 'expand' not in k:
            current = v.summarize_proof()
            for i in range(len(current)):
                if labels[current[i]][0] == '$p':
                    proof_count_list.append(current[i])
    count_dict = dict(Counter(proof_count_list))
    return count_dict


def analyze_predictions_test(predictions, labels, word_dict, mm, dataset_proof_names, is_expanded, output_directory, save_mode):
    # need prediction, label list, mm_proof format, as well as the raw dataset
    # when it's unexpanded, correct means color nothing
    mm_proof_labels = mm.proofs

    num_is_tree = 0
    num_correct = 0
    num_color_one_or_less = 0
    num_color_all = 0
    num_meaningful = 0  # do not count the correct ones
    counter = 0

    for name in dataset_proof_names:
        if is_expanded:
            variant = int(name.split('_')[-1])
            proof_name = name[:name.find('variant') - 1]
            proof_label = mm_proof_labels[proof_name][variant]
        else:
            proof_label = mm_proof_labels[name]
        proof_prediction = copy.deepcopy(proof_label)
        proof_label.name = name + '_label'
        proof_prediction.name = name + '_prediction'
        proof_raw = export_single_new(proof_label, word_dict, allow_update=False)
        proof_raw.insert(0, name)

        proof_length = len(proof_raw[3])

        current_labels = labels[counter:counter + proof_length]

        num_colored_nodes = np.sum(np.round(current_labels))
        color_one_or_less = int(num_colored_nodes <= 1)
        if color_one_or_less:
            print(name)
            assert 0 == 1
        num_color_one_or_less += color_one_or_less
        color_all = int(num_colored_nodes == proof_length)
        num_color_all += color_all

        color_proof_tree(proof_label, current_labels)
        correct = check_proof_correct(current_labels, current_labels)
        is_tree = check_proof_is_tree(proof_raw, current_labels)

        if correct == 1 and is_tree != 1 and not color_one_or_less:
            raise NotImplementedError('if correct, should definitely be a tree')
        # do extraction
        if is_tree == 1 and not color_one_or_less:
            if is_expanded:
                extracted_proof_name = name.replace('expand_', 'extracted_')
                if not correct:
                    extracted_proof_name = 'new_theorem_{0}_from_'.format(num_meaningful) + extracted_proof_name
            else:
                extracted_proof_name = 'new_theorem_{0}_from_'.format(num_meaningful) + name
            meaningful_proof = check_proof_meaningful(mm, proof_label, extracted_proof_name)
            if correct == 1 and meaningful_proof is None:
                print('correct proof should definitely be meaningful')
            if meaningful_proof is not None and correct == 0:
                num_meaningful += 1
                if save_mode == 'meaningful':
                    meaningful_proof.draw_graph_2(output_dir=output_directory)
                    proof_prediction.draw_graph_2(output_dir=output_directory)
                    proof_label.draw_graph_2(output_dir=output_directory)
        num_correct += correct
        num_is_tree += is_tree
        if save_mode == 'all':
            proof_prediction.draw_graph_2(output_dir=output_directory)
            proof_label.draw_graph_2(output_dir=output_directory)
        counter += proof_length
    print('num correct: {0}'.format(num_correct))
    print('num color one or less: {0}'.format(num_color_one_or_less))
    print('num color all: {0}'.format(num_color_all))
    print('num meaningful but not correct: {0}'.format(num_meaningful))
    print('num meaningful: {0}'.format(num_meaningful + num_correct))
    print('num is_tree: {0}'.format(num_is_tree))
    print('num total: {0}'.format(len(dataset_proof_names)))


def analyze_predictions(predictions, labels, word_dict, mm, dataset_proof_names, is_expanded, output_directory, save_mode):
    # need prediction, label list, mm_proof format, as well as the raw dataset
    # when it's unexpanded, correct means color nothing
    mm_proof_labels = mm.proofs

    num_is_tree = 0
    num_correct = 0
    num_color_one_or_less = 0
    num_color_all = 0
    num_meaningful = 0  # do not count the correct ones
    counter = 0
    new_theorems = []
    for name in dataset_proof_names:
        if is_expanded:
            variant = int(name.split('_')[-1])
            proof_name = name[:name.find('variant') - 1]
            proof_label = mm_proof_labels[proof_name][variant]
        else:
            proof_label = mm_proof_labels[name]
        proof_prediction = copy.deepcopy(proof_label)
        proof_label.name = name + '_label'
        proof_prediction.name = name + '_prediction'
        proof_raw = export_single_new(proof_label, word_dict, allow_update=False)
        proof_raw.insert(0, name)

        proof_length = len(proof_raw[3])
        if not is_expanded and proof_length > args.unexpanded_node_limit_proof:
            continue
            # since we have not recorded its prediction

        current_predictions = predictions[counter:counter + proof_length]
        current_labels = labels[counter:counter + proof_length]

        num_colored_nodes = np.sum(np.round(current_predictions))
        color_one_or_less = int(num_colored_nodes <= 1)
        num_color_one_or_less += color_one_or_less
        color_all = int(num_colored_nodes == proof_length)
        num_color_all += color_all

        color_proof_tree(proof_prediction, current_predictions)
        correct = check_proof_correct(current_predictions, current_labels)
        is_tree = check_proof_is_tree(proof_raw, current_predictions)

        if correct == 1 and is_tree != 1 and not color_one_or_less:
            raise NotImplementedError('if correct, should definitely be a tree')
        # do extraction
        if is_tree == 1 and not color_one_or_less:
            if is_expanded:
                extracted_proof_name = name.replace('expand_', 'extracted_')
                if not correct:
                    extracted_proof_name = 'new_theorem_{0}_from_'.format(num_meaningful) + extracted_proof_name
            else:
                extracted_proof_name = 'new_theorem_{0}_from_'.format(num_meaningful) + name
            meaningful_proof = check_proof_meaningful(mm, proof_prediction, extracted_proof_name)
            if correct == 1 and meaningful_proof is None:
                ground_truth_verifiable = check_proof_meaningful(mm, proof_label, extracted_proof_name)
                if ground_truth_verifiable:
                    raise NotImplementedError('correct and ground truth verifiable proof should definitely be meaningful')
                else:
                    print('cannot verify due to ground truth not even verifiable')
            if meaningful_proof is not None and correct == 0:
                num_meaningful += 1
                new_theorems.append(meaningful_proof)
                if save_mode == 'meaningful':
                    meaningful_proof.draw_graph_2(output_dir=output_directory)
        num_correct += correct
        num_is_tree += is_tree
        if save_mode == 'all':
            proof_prediction.draw_graph_2(output_dir=output_directory)
            proof_label.draw_graph_2(output_dir=output_directory)
        counter += proof_length
    if 'new_theorems' not in mm.proofs.keys():
        mm.proofs['new_theorems'] = new_theorems
    else:
        mm.proofs['new_theorems'].extend(new_theorems)
    print('num correct: {0}'.format(num_correct))
    print('num color one or less: {0}'.format(num_color_one_or_less))
    print('num color all: {0}'.format(num_color_all))
    print('num meaningful but not correct: {0}'.format(num_meaningful))
    print('num meaningful: {0}'.format(num_meaningful + num_correct))
    print('num is_tree: {0}'.format(num_is_tree))
    print('num total: {0}'.format(len(dataset_proof_names)))


def get_dataset_proof_names(mm_proofs, analyze_data, path):
    if analyze_data != 'unexpanded':
        if os.path.isfile(path + '{0}_proof_names.pkl'.format(analyze_data)):
            print('loading proof names')
            with open(path + '{0}_proof_names.pkl'.format(analyze_data), 'rb') as f:
                names = pickle.load(f)
        else:
            print('generating proof names')
            with open(path + '{0}_dataset.pkl'.format(analyze_data), 'rb') as f:
                dataset_raw = pickle.load(f)
            names = [e[0] for e in dataset_raw]
            del dataset_raw
            print('done generating proof names')
            with open(path + '{0}_proof_names.pkl'.format(analyze_data), 'wb') as f:
                pickle.dump(names, f)
    else:
        names = []
        for k in mm_proofs.keys():
            if 'expand_' not in k and 'new_theorems' not in k:
                names.append(k)
    return names


def is_similar(proof_1, proof_2, include_sub):
    if not include_sub:
        return proof_1 == proof_2
    else:
        if len(proof_1) != len(proof_2):
            return False
        else:
            for i in range(len(proof_1)):
                if proof_1[i] != proof_2[i]:
                    if proof_1[i] in ['sub0', 'sub1', 'sub2'] and proof_2[i] != '' or proof_2[i] in['sub0', 'sub1', 'sub2'] and proof_1[i] != '':
                        pass
                    else:
                        return False
        return True


def remove_redundancy_from_list(proof_list, labels, index):
    # check index + 1 but not index
    summary_list = []
    for proof_node in proof_list:
        summary = proof_node.summarize_proof()
        for i in range(len(summary)):
            node = summary[i]
            if labels[node][0] in ['$f', '$e']:
                summary[i] = ''
        summary_list.append(tuple(summary))
    delete_list = []
    for i in range(len(summary_list) - 1, index, -1):
        assert len(summary_list[i]) > 1
        for j in range(i - 1, -1, -1):
            if is_similar(summary_list[i], summary_list[j], True):
                print('{0} is similar to {1}'.format(proof_list[i].name, proof_list[j].name))
                delete_list.append(i)
                break
    return delete_list


def remove_redundancy(mm):
    # remove redundant proofs from original and within new theorems
    new_theorems = mm.proofs['new_theorems']
    original_theorems = []
    for k, v in mm.proofs.items():
        if 'expand_' not in k and k != 'new_theorems':
            assert type(v) != list
            original_theorems.append(v)
    delete_list = remove_redundancy_from_list(original_theorems + new_theorems, mm.labels, len(original_theorems) - 1)
    delete_list = [e - len(original_theorems) for e in delete_list]
    mm.proofs['new_theorems'] = [new_theorems[i] for i in range(len(new_theorems)) if i not in delete_list]


def get_dvs(proof, labels):
    # custom dfs
    visited = []
    a = [proof]
    while len(a) > 0:
        node = a.pop(0)
        visited.append(node)
        temp = []
        for child in node.mand_vars:
            temp.append(child)
        for child in node.hps:
            temp.append(child)
        a = temp + a
    all_dvs = []
    for node in visited:
        if node.find_min_height() == 2:
            if node.type in ['$a', '$p']:
                mand_var_dict = {}
                _, (dvs, mand_vars, _, _) = labels[node.label]
                for mand_var in list(mand_vars):
                    assert mand_var not in mand_var_dict
                    mand_var_dict[mand_var[1]] = len(mand_var_dict)
                dvs = list(dvs)
                replaced_dvs = []
                for i in range(len(dvs)):
                    replaced_dvs.append([])
                    for j in range(len(dvs[i])):
                        replaced_dvs[i].append(mand_var_dict[dvs[i][j]])
                actual_mand_vars = node.mand_vars
                for dvs in replaced_dvs:
                    product_list = []
                    for index in dvs:
                        # actual_dvs.append(actual_mand_vars[index].expr[1])
                        mand_var_node = actual_mand_vars[index]
                        if len(mand_var_node.mand_vars) == 0:
                            assert len(mand_var_node.expr) == 2
                            product_list.append([actual_mand_vars[index].expr[1]])
                        else:
                            leaves = mand_var_node.get_leaves()
                            current = []
                            for leaf in leaves:
                                if leaf.type == '$f':
                                    current.append(leaf.expr[1])
                            product_list.append(list(set(current)))
                    actual_dvs = list(itertools.product(*product_list))
                    sorted_actual_dvs = [tuple(sorted(list(e))) for e in actual_dvs]
                    for e in sorted_actual_dvs:
                        if e not in all_dvs:
                            if len(set(e)) != 1:
                                all_dvs.append(e)
                        else:
                            pass
    return all_dvs


def export_single_new_theorem(proof, labels):
    out = '  ${\n'
    all_dvs = get_dvs(proof, labels)
    for dvs in all_dvs:
        elements = ['$d'] + list(dvs)
        elements.append('$.\n')
        out += '    ' + ' '.join(elements)
    leaves = proof.get_leaves()
    for leaf in leaves:
        if leaf.type == '$e':
            elements = []
            elements.append(leaf.label)
            elements.append('$e')
            elements += leaf.expr
            elements.append('$.\n')
            out += '    ' + ' '.join(elements)
    # write proof expression
    res = [proof.name, '$p'] + proof.expr
    res.append('$=')
    for e in proof.summarize_proof():
        res.append(e)
    res.append('$.\n')
    out += '    ' + ' '.join(res)
    out += '  $}\n\n'
    return out


def export_new_theorems(file_path, mm):
    if not os.path.isfile(file_path):
        copyfile('set.mm', file_path + 'augmented_set.mm')
    output = []
    print('before dvs: {0}'.format(len(mm.proofs['new_theorems'])))
    for proof in mm.proofs['new_theorems']:
        res = export_single_new_theorem(proof, mm.labels)
        if res != '':
            output.append(res)
    print('after dvs: {0}'.format(len(output)))
    output = ''.join(output)
    with open(file_path + 'augmented_set.mm', 'a') as f:
        f.write(output)


def main(args):
    with open(args.path + 'args.json', 'r') as f:
        arg_dict = json.load(f)
    print(arg_dict)
    if 'average' not in arg_dict:
        arg_dict['average'] = 'node'
    checkpoint_path = arg_dict['checkpoint_directory']
    if args.checkpoint_path != '':
        checkpoint_path = args.checkpoint_path
    model = model_names_dict[arg_dict['model']].load_from_checkpoint(checkpoint_path + 'epoch={0}.ckpt'.format(args.epoch), **arg_dict)
    if args.gpu:
        model = model.cuda()
    model.eval()
    hparams = model.hparams
    data_path = hparams['data_path']
    if args.data_path != '':
        data_path = args.data_path
    if 'validation_path' not in arg_dict:
        validation_path = data_path
    else:
        validation_path = arg_dict['validation_path']
    if args.validation_path != '':
        validation_path = args.validation_path
    with open('{0}word_dict.pkl'.format(data_path), 'rb') as f:
        word_dict = pickle.load(f)
    max_length = hparams['max_length']
    batch_size = args.batch_size
    direction = hparams['direction']
    save_directory = args.save_directory
    figure_save_directory = save_directory + args.figure_save_directory
    save_mode = args.save_mode
    all_analyze_data = args.analyze_data
    num_nodes_limit_per_batch = args.num_nodes_limit_per_batch
    mm_path = args.mm_path
    gpu = args.gpu
    if not os.path.isdir(args.save_directory):
        os.makedirs(args.save_directory)
    if not os.path.isdir(figure_save_directory):
        os.makedirs(figure_save_directory)
    if 'use_node_information' not in hparams:
        use_node_information = 'strnode'
    else:
        use_node_information = hparams['use_node_information']
    with open(mm_path + 'dataset_and_raw_mm.pkl', 'rb') as f:
        mm = pickle.load(f)
    with open(mm_path + 'in_scope_labels.pkl', 'rb') as f:
        in_scope_labels = pickle.load(f)
    # with open(mm_path + 'evaluate_raw_mm.pkl', 'rb') as f:
    #     mm = pickle.load(f)
    mm.in_scope_labels = in_scope_labels

    all_analyze_data = all_analyze_data.split('_')
    for analyze_data in all_analyze_data:
        if analyze_data == 'train':
            loader_path = data_path
        elif analyze_data == 'valid':
            loader_path = validation_path
        elif analyze_data == 'unexpanded':
            loader_path = mm_path
        elif analyze_data == 'test':
            loader_path = validation_path
        else:
            raise NotImplementedError
        if not os.path.isfile(save_directory + '{0}_prediction.npy'.format(analyze_data)):
            print('getting predictions and labels for {0}'.format(analyze_data))
            if analyze_data == 'unexpanded':
                loader = get_data(loader_path, analyze_data, word_dict, max_length, 1, direction,
                                  use_node_information, num_workers=0, shuffle=False, partial=1.0,
                                  num_nodes_limit_per_batch=num_nodes_limit_per_batch)
            else:
                loader = get_data(loader_path, analyze_data, word_dict, max_length, batch_size, direction, use_node_information, num_workers=0, shuffle=False, partial=1.0, num_nodes_limit_per_batch=num_nodes_limit_per_batch)
            predictions, labels = evaluate_loader(loader, model, gpu)
            del loader
            np.save(save_directory + '{0}_prediction.npy'.format(analyze_data), predictions)
            np.save(save_directory + '{0}_label.npy'.format(analyze_data), labels)
        else:
            print('loading predictions and labels for {0}'.format(analyze_data))
            predictions = np.load(save_directory + '{0}_prediction.npy'.format(analyze_data))
            labels = np.load(save_directory + '{0}_label.npy'.format(analyze_data))

        # count_proof_name_frequency(mm)
        dataset_proof_names = get_dataset_proof_names(mm.proofs, analyze_data, loader_path)
        analyze_predictions(predictions, labels, word_dict, mm, dataset_proof_names, analyze_data != 'unexpanded', figure_save_directory, save_mode)
        # test for labels
    print('total raw new theorems is {0}'.format(len(mm.proofs['new_theorems'])))
    with open(save_directory + 'raw_augmented_mm.pkl', 'wb') as f:
        pickle.dump(mm, f)
    remove_redundancy(mm)
    print('total actual new theorems is {0}'.format(len(mm.proofs['new_theorems'])))
    with open(save_directory + 'actual_augmented_mm.pkl', 'wb') as f:
        pickle.dump(mm, f)

    export_new_theorems(save_directory, mm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluate model")
    parser.add_argument('-e', dest='epoch', type=int, default=500)
    parser.add_argument('-path', dest='path', type=str, default='')
    parser.add_argument('-s', dest='save_directory', type=str, default='')
    parser.add_argument('-fs', dest='figure_save_directory', type=str, default='extracted/')
    parser.add_argument('-cp', dest='checkpoint_path', type=str, default='')
    parser.add_argument('-dp', dest='data_path', type=str, default='')
    parser.add_argument('-vp', dest='validation_path', type=str, default='')
    parser.add_argument('-sm', dest='save_mode', type=str, default='')
    parser.add_argument('-ad', dest='analyze_data', type=str, default='valid')
    parser.add_argument('-b', dest='batch_size', type=int, default=16)
    parser.add_argument('-node_limit', dest='num_nodes_limit_per_batch', type=int, default=-1)
    parser.add_argument('-unl', dest='unexpanded_node_limit_proof', type=int, default=5000)  # prevent coloring very large raw proofs, keep it here
    parser.add_argument('-mp', dest='mm_path', type=str, default='dataset/propositional_mm/')
    parser.add_argument('-g', dest='gpu', type=int, default=1)
    args = parser.parse_args()
    main(args)
