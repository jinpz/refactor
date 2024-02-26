from theorem_expansion import *
from theorem_verification import export_single_new_theorem
from shutil import copyfile


def follow_color(uncolored_node, colored_node):
    uncolored_node.subst = colored_node.subst
    if len(colored_node.mand_vars) > 0:
        for v1, v2 in zip(uncolored_node.mand_vars, colored_node.mand_vars):
            follow_color(v1, v2)
    if len(colored_node.hps) > 0:
        for v1, v2 in zip(uncolored_node.hps, colored_node.hps):
            follow_color(v1, v2)


def get_dfs(proof_node):
    # custom dfs
    visited = []
    a = [proof_node]
    while len(a) > 0:
        node = a.pop(0)
        visited.append(node)
        temp = []
        for child in node.mand_vars:
            temp.append(child)
        for child in node.hps:
            temp.append(child)
        a = temp + a
    return visited


def get_post_order(proof_node):
    # post order traversal
    res = []
    for var in proof_node.mand_vars:
        res += get_post_order(var)
    for hp in proof_node.hps:
        res += get_post_order(hp)
    res.append(proof_node)
    return res


def refactor_proof_single(original_node, matching_proof, labels):
    # assume original node is the root that matches matching theorem
    assert not any([e.subst for e in get_dfs(original_node)])
    assert not any([e.subst for e in get_dfs(matching_proof)])
    leaves = matching_proof.get_leaves()
    # color $e and $f leaves only
    for i in range(len(leaves)):
        if leaves[i].type in ['$e', '$f']:
            leaves[i].subst = True
        else:
            # where $a leaves go
            pass
    # only keep $e and $f
    leaves = [e for e in leaves if e.type in ['$e', '$f']]
    follow_color(original_node, matching_proof)
    visited = get_post_order(original_node)
    colored_leaves = [e for e in visited if e.subst is True]
    assert len(colored_leaves) == len(leaves)
    # use expression for uniqueness, assume hypothesis is one to one
    num_mand_vars = len(labels[matching_proof.name][1][1])
    num_hps = len(labels[matching_proof.name][1][2])
    matching_mand_vars = list(labels[matching_proof.name][1][1])
    assert len(set(matching_mand_vars)) == len(matching_mand_vars)
    leaves_mand_vars_indices = [i for i in range(len(leaves)) if leaves[i].type == '$f']
    leaves_mand_vars = [leaves[i] for i in leaves_mand_vars_indices]
    leaves_hps_indices = [i for i in range(len(leaves)) if leaves[i].type == '$e']
    assert len(leaves_hps_indices) == num_hps
    unique_index_bucket = []
    for i in range(num_mand_vars):
        unique_index_bucket.append([])
    for i in range(len(leaves) - num_hps):
        if tuple(leaves_mand_vars[i].expr) in matching_mand_vars:
            index = matching_mand_vars.index(tuple(leaves_mand_vars[i].expr))
            unique_index_bucket[index].append(leaves_mand_vars_indices[i])
        else:
            # here is where the temporary $f goes
            pass
    # check uniqueness
    for i in range(len(unique_index_bucket)):
        bucket = unique_index_bucket[i]
        expressions = [tuple(colored_leaves[e].expr) for e in bucket]
        assert len(set(expressions)) == 1
    needed_leaves = []
    for i in range(len(unique_index_bucket)):
        bucket = unique_index_bucket[i]
        # can simply use the first one
        needed_leaves.append(colored_leaves[bucket[0]])
    for i in range(num_hps):
        needed_leaves.append(colored_leaves[leaves_hps_indices[i]])

    new_mand_vars = []
    # uncolor it
    for i in range(len(colored_leaves)):
        colored_leaves[i].subst = False
    for i in range(num_mand_vars):
        # needed_leaves[i].subst = False
        new_mand_vars.append(needed_leaves[i])
    new_hps = []
    for i in range(num_hps):
        # needed_leaves[num_mand_vars + i].subst = False
        new_hps.append(needed_leaves[num_mand_vars + i])
    original_node.mand_vars = new_mand_vars
    original_node.hps = new_hps
    original_node.label = matching_proof.name
    original_node.type = '$p'
    for i in range(len(leaves)):
        leaves[i].subst = False


def additional_check(original_node, matching_proof, labels):
    res = True
    # assume original node is the root that matches matching theorem
    assert not any([e.subst for e in get_dfs(original_node)])
    assert not any([e.subst for e in get_dfs(matching_proof)])
    leaves = matching_proof.get_leaves()
    # color $e and $f leaves only
    for i in range(len(leaves)):
        if leaves[i].type in ['$e', '$f']:
            leaves[i].subst = True
        else:
            # where $a leaves go
            pass
    # only keep $e and $f
    leaves = [e for e in leaves if e.type in ['$e', '$f']]
    follow_color(original_node, matching_proof)
    visited = get_post_order(original_node)
    colored_leaves = [e for e in visited if e.subst is True]
    assert len(colored_leaves) == len(leaves)
    # use expression for uniqueness, assume hypothesis is one to one
    num_mand_vars = len(labels[matching_proof.name][1][1])
    num_hps = len(labels[matching_proof.name][1][2])
    matching_mand_vars = list(labels[matching_proof.name][1][1])
    assert len(set(matching_mand_vars)) == len(matching_mand_vars)
    leaves_mand_vars_indices = [i for i in range(len(leaves)) if leaves[i].type == '$f']
    leaves_mand_vars = [leaves[i] for i in leaves_mand_vars_indices]
    leaves_hps_indices = [i for i in range(len(leaves)) if leaves[i].type == '$e']
    assert len(leaves_hps_indices) == num_hps
    unique_index_bucket = []
    for i in range(num_mand_vars):
        unique_index_bucket.append([])
    for i in range(len(leaves) - num_hps):
        if tuple(leaves_mand_vars[i].expr) in matching_mand_vars:
            index = matching_mand_vars.index(tuple(leaves_mand_vars[i].expr))
            unique_index_bucket[index].append(leaves_mand_vars_indices[i])
        else:
            # here is where the temporary $f goes
            pass
    # check uniqueness
    for i in range(len(unique_index_bucket)):
        bucket = unique_index_bucket[i]
        expressions = [tuple(colored_leaves[e].expr) for e in bucket]
        if len(set(expressions)) != 1:
            res = False
            break
    # uncolor it
    for i in range(len(colored_leaves)):
        colored_leaves[i].subst = False
    for i in range(len(leaves)):
        leaves[i].subst = False
    return res


def match_theorem_current_node(original_node, matching_node, labels, counter):
    if labels[matching_node.label][0] in ['$a', '$p']:
        if original_node.label != matching_node.label:
            return None
        else:
            assert len(matching_node.mand_vars) == len(original_node.mand_vars)  # should be true
            assert len(matching_node.hps) == len(original_node.hps)  # should be true
            for i in range(len(matching_node.mand_vars)):
                match_res = match_theorem_current_node(original_node.mand_vars[i], matching_node.mand_vars[i], labels, counter - 1)
                if match_res is None:
                    return None
            for i in range(len(matching_node.hps)):
                match_res = match_theorem_current_node(original_node.hps[i], matching_node.hps[i], labels, counter - 1)
                if match_res is None:
                    return None
            if counter == 0:
                check_res = additional_check(original_node, matching_node, labels)
                if not check_res:
                    return None
            return original_node

    elif labels[matching_node.label][0] in ['$e', '$f']:
        assert len(matching_node.mand_vars) == 0
        assert len(matching_node.hps) == 0
        return original_node
    else:
        raise NotImplementedError('unseen label')


def refactor_all(mm):
    all_proofs = mm.proofs
    all_labels = mm.labels
    all_original_proofs = [v for k, v in all_proofs.items() if 'new_theorem' not in k]
    all_new_proofs = [v for k, v in all_proofs.items() if 'new_theorem' in k]
    # another filtering
    all_new_proofs = [e for e in all_new_proofs if e.find_max_height() > 2]
    all_new_proofs = sorted(all_new_proofs, key=lambda x: len(x.summarize_proof()))
    refactor_counts = np.zeros((len(all_original_proofs), len(all_new_proofs)))
    refactored_theorems = []
    for i in range(len(all_original_proofs)):
        refactored_proof = copy.deepcopy(all_original_proofs[i])  # placeholder for new proof
        refactored_proof.name = 'refactored_' + refactored_proof.name
        for j in range(len(all_new_proofs)):
            new_proof = all_new_proofs[j]
            current_proof_finish_refactor = False
            while not current_proof_finish_refactor:
                refactored_proof_list = get_post_order(refactored_proof)
                finish_flag = True
                for k in range(len(refactored_proof_list)):
                    node = refactored_proof_list[k]
                    match_res = match_theorem_current_node(node, new_proof, all_labels, 0)
                    if match_res is not None:
                        # try to refactor, first backup in case unsuccessful
                        refactor_proof_single(node, new_proof, all_labels)
                        verified, _ = mm.verify_custom(refactored_proof.expr, refactored_proof.summarize_proof(),
                                                       name='', mode='other')
                        if not verified:
                            print('only subtree pattern match, still cannot refactor')
                            # restore the refactored proof before this attempt
                            raise NotImplementedError('failed to verify i = {0}, j = {1}'.format(i, j))
                        else:
                            finish_flag = False
                            refactor_counts[i, j] += 1
                            break
                if finish_flag:
                    # current proof cannot be refactored by this new theorem
                    current_proof_finish_refactor = True
        if np.sum(refactor_counts[i, :]) > 0:
            refactored_theorems.append(refactored_proof)
    print('total refactor operations: {0}'.format(refactor_counts.sum()))
    print('total proofs refactored: {0}'.format(len(refactored_theorems)))
    return refactored_theorems


def main(args):
    if os.path.isfile(args.path + 'augmented_set_mm.pkl'):
        with open(args.path + 'augmented_set_mm.pkl', 'rb') as f:
            mm = pickle.load(f)
    else:
        mm = MM(0, -1)
        with open(args.path + args.main_file, 'r') as f:
            mm.read(toks(f))
        with open(args.path + 'augmented_set_mm.pkl', 'wb') as f:
            pickle.dump(mm, f)

    refactored_theorems = refactor_all(mm)
    output = []
    for e in refactored_theorems:
        output.append(export_single_new_theorem(e, mm.labels))
    output = ''.join(output)
    copyfile(args.path + args.main_file, args.path + args.output_name)
    with open(args.path + args.output_name, 'a') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluate model")
    parser.add_argument('-p', dest='path', type=str, default='')
    parser.add_argument('-m', dest='main_file', type=str, default='augmented_set.mm')
    parser.add_argument('-o', dest='output_name', type=str, default='augmented_set_refactored.mm')
    args = parser.parse_args()
    main(args)
