#!/usr/bin/env python3
# mmverify.py -- Proof verifier for the Metamath language
# Copyright (C) 2002 Raph Levien raph (at) acm (dot) org
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License

# To run the program, type
#   $ python3 mmverify.py < set.mm 2> set.log
# and set.log will have the verification results.

# (nm 27-Jun-2005) mmverify.py requires that a $f hypothesis must not occur
# after a $e hypothesis in the same scope, even though this is allowed by
# the Metamath spec.  This is not a serious limitation since it can be
# met by rearranging the hypothesis order.
# (rl 2-Oct-2006) removed extraneous line found by Jason Orendorff
# (sf 27-Jan-2013) ported to Python 3, added support for compressed proofs
# and file inclusion

import sys
import itertools
import collections
from collections import defaultdict
import os.path
import copy
from graphviz import Digraph
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import pickle
import argparse
import pandas as pd
import time
from scipy.stats import entropy

verbosity = 1


class MMError(Exception): pass


class MMKeyError(MMError, KeyError): pass


def vprint(vlevel, *args):
    if verbosity >= vlevel: print(*args, file=sys.stderr)


class toks:
    def __init__(self, lines):
        self.lines_buf = [lines]
        self.tokbuf = []
        self.imported_files = set()

    def read(self):  # read one thing at a time separated by space, can read between lines
        while self.tokbuf == []:
            line = self.lines_buf[-1].readline()  # read a line after finishing the last one
            if not line:
                self.lines_buf.pop().close()
                if not self.lines_buf: return None
            else:
                self.tokbuf = line.split()  # split a line by space
                self.tokbuf.reverse()  # basically remove element from left to right
        return self.tokbuf.pop()

    def readf(self):  # read (possibly) include files
        tok = self.read()
        while tok == '$[':
            filename = self.read()
            endbracket = self.read()
            if endbracket != '$]':
                raise MMError('Incusion command not terminated')
            filename = os.path.realpath(filename)
            if filename not in self.imported_files:
                self.lines_buf.append(open(filename, 'r'))
                self.imported_files.add(filename)
            tok = self.read()
        return tok

    def readc(
            self):  # read next word until something that is not in a comment (i.e. not $( or $) or anything between the two)
        while 1:
            tok = self.readf()
            if tok is None: return None
            if tok == '$(':
                while tok != '$)':
                    tok = self.read()
            else:
                return tok

    def readstat(self):  # read statement: read actual content between $X and $. (can span multiple lines)
        stat = []
        tok = self.readc()
        while tok != '$.':
            if tok is None: raise MMError('EOF before $.')
            stat.append(tok)
            tok = self.readc()
        return stat


class Frame:
    def __init__(self):
        self.c = set()
        self.v = set()
        self.d = set()
        self.f = []
        self.f_labels = {}
        self.e = []
        self.e_labels = {}


class FrameStack(list):
    def push(self):
        self.append(Frame())

    def add_c(self, tok):  # add constant to constant set in a frame
        frame = self[-1]
        if tok in frame.c: raise MMError('const already defined in scope')
        if tok in frame.v:
            raise MMError('const already defined as var in scope')
        frame.c.add(tok)

    def add_v(self, tok):
        frame = self[-1]
        if tok in frame.v: raise MMError('var already defined in scope')
        if tok in frame.c:
            raise MMError('var already defined as const in scope')
        frame.v.add(tok)

    def add_f(self, var, kind, label):
        if not self.lookup_v(var):
            raise MMError('var in $f not defined: {0}'.format(var))
        if not self.lookup_c(kind):
            raise MMError('const in $f not defined {0}'.format(kind))
        frame = self[-1]
        if var in frame.f_labels.keys():
            raise MMError('var in $f already defined in scope')
        frame.f.append((var, kind))
        frame.f_labels[var] = label

    def add_e(self, stat, label):
        frame = self[-1]
        frame.e.append(stat)
        frame.e_labels[tuple(stat)] = label

    def add_d(self, stat):
        frame = self[-1]
        frame.d.update(((min(x, y), max(x, y))  # enforce an order
                        for x, y in itertools.product(stat, stat) if x != y))

    def lookup_c(self, tok):
        return any((tok in fr.c for fr in reversed(self)))

    def lookup_v(self, tok):
        return any((tok in fr.v for fr in reversed(self)))

    def lookup_f(self, var):  # return floating hypothesis label
        for frame in reversed(self):  # look at inner scope first
            try:
                return frame.f_labels[var]
            except KeyError:
                pass
        raise MMKeyError(var)

    def lookup_d(self, x, y):
        return any(((min(x, y), max(x, y)) in fr.d for fr in
                    reversed(self)))  # see if outer scope or this scope has disjoint restriction

    def lookup_e(self, stmt):
        stmt_t = tuple(stmt)
        for frame in reversed(self):
            try:
                return frame.e_labels[stmt_t]
            except KeyError:
                pass
        raise MMKeyError(stmt_t)

    def make_assertion(self, stat):
        frame = self[-1]
        e_hyps = [eh for fr in self for eh in fr.e]  # essential hypothesis
        mand_vars = {tok for hyp in itertools.chain(e_hyps, [stat])
                     for tok in hyp if self.lookup_v(tok)}  # mandatory variables from e_hyps and statement

        dvs = {(x, y) for fr in self for (x, y) in
               fr.d.intersection(itertools.product(mand_vars, mand_vars))}  # disjoint variable set

        f_hyps = collections.deque()  # floating hypothesis
        for fr in reversed(self):
            for v, k in reversed(fr.f):  # RPN notation
                if v in mand_vars:
                    f_hyps.appendleft((k, v))
                    mand_vars.remove(v)

        vprint(18, 'ma:', (dvs, f_hyps, e_hyps, stat))
        return dvs, f_hyps, e_hyps, stat


class ProofNode:

    def __init__(self, label, type, data):
        self.label = label
        self.type = type  # node type e or f should not have any mand_vars or hps
        self.data = data
        self.name = ""
        self.expr = ""
        self.mand_vars = []
        self.hps = []
        self.subst = False

    @property
    def str(self):
        return "".join(self.expr)

    def set_expr(self, expr):
        self.expr = expr

    def set_name(self, name):
        self.name = name

    def add_mand_vars(self, mand_var):
        assert self.type in '$p' or self.type in '$a'
        self.mand_vars.append(mand_var)

    def add_hps(self, hp):
        assert self.type in '$p' or self.type in '$a'
        self.hps.append(hp)

    def summarize_proof(self):
        proof = []
        # recursive expansion
        for var in self.mand_vars:
            proof += var.summarize_proof()
        for hp in self.hps:
            proof += hp.summarize_proof()
        proof.append(self.label)
        return proof

    def find_max_height(self):
        if len(self.mand_vars) == 0 and len(self.hps) == 0:
            return 1
        else:
            max_height = -1
            for mand_var in self.mand_vars:
                current_max_height = mand_var.find_max_height()
                if current_max_height > max_height:
                    max_height = current_max_height
            for hp in self.hps:
                current_max_height = hp.find_max_height()
                if current_max_height > max_height:
                    max_height = current_max_height
            return max_height + 1

    def find_min_height(self):
        if len(self.mand_vars) == 0 and len(self.hps) == 0:
            return 1
        else:
            min_height = float('inf')
            for mand_var in self.mand_vars:
                current_min_height = mand_var.find_min_height()
                if current_min_height < min_height:
                    min_height = current_min_height
            for hp in self.hps:
                current_min_height = hp.find_min_height()
                if current_min_height < min_height:
                    min_height = current_min_height
            return min_height + 1

    def expand_proof(self, subst):  # expand proof and also color expand node
        if len(self.mand_vars) > 0:
            for i in range(len(self.mand_vars)):
                if self.mand_vars[i].mand_vars == [] and self.mand_vars[i].hps == []:
                    current_key = tuple(self.mand_vars[i].expr)
                    if current_key in subst.keys():
                        subst_node = copy.deepcopy(subst[current_key])
                        subst_node.subst = True
                        self.mand_vars[i] = subst_node
                else:
                    self.mand_vars[i].expand_proof(subst)

        if len(self.hps) > 0:
            for i in range(len(self.hps)):
                if self.hps[i].mand_vars == [] and self.hps[i].hps == []:
                    current_key = tuple(self.hps[i].expr)
                    if tuple(self.hps[i].expr) in subst.keys():
                        subst_node = copy.deepcopy(subst[current_key])
                        subst_node.subst = True
                        self.hps[i] = subst_node
                else:
                    self.hps[i].expand_proof(subst)

    def get_leaves(self, change_type=False):  # need to substitute expression, data, label
        # change type will force change mand_vars type to $f and hps to $e
        leaves = []
        for i in range(len(self.mand_vars)):
            if self.mand_vars[i].mand_vars == [] and self.mand_vars[i].hps == []:
                if change_type:
                    self.mand_vars[i].type = '$f'
                leaves.append(self.mand_vars[i])
            else:
                res = self.mand_vars[i].get_leaves(change_type=change_type)
                if res is not None:
                    leaves.extend(res)
        for i in range(len(self.hps)):
            if self.hps[i].mand_vars == [] and self.hps[i].hps == []:
                if change_type:
                    self.hps[i].type = '$e'
                leaves.append(self.hps[i])
            else:
                res = self.hps[i].get_leaves(change_type=change_type)
                if res is not None:
                    leaves.extend(res)
        return leaves

    def color_all(self):
        self.subst = True
        for mand_var in self.mand_vars:
            mand_var.color_all()
        for hp in self.hps:
            hp.color_all()

    def mark_subst(self, node, propagate=True):  # follow the color of expand node
        self.subst = node.subst
        if len(self.mand_vars) > 0:
            for v1, v2 in zip(self.mand_vars, node.mand_vars):
                v1.mark_subst(v2)

        if len(self.hps) > 0:
            for h1, h2 in zip(self.hps, node.hps):
                h1.mark_subst(h2)

        list_subst = [v.subst for v in self.mand_vars] + [h.subst for h in
                                                          self.hps]  # propagate according to immediate child
        if propagate and len(list_subst) > 0 and any(list_subst):
            self.subst = True
            for v in self.mand_vars:
                v.subst = True
            for h in self.hps:
                h.subst = True

    def mark_subst_old(self, node, propagate=True):  # follow the color of expand node
        self.subst = node.subst
        if len(self.mand_vars) > 0:
            for v1, v2 in zip(self.mand_vars, node.mand_vars):
                v1.mark_subst_old(v2)

        if len(self.hps) > 0:
            for h1, h2 in zip(self.hps, node.hps):
                h1.mark_subst_old(h2)

        list_subst = [v.subst for v in self.mand_vars] + [h.subst for h in
                                                          self.hps]  # propagate according to immediate child
        if propagate and len(list_subst) > 0 and all(list_subst):
            self.subst = True

    def copy_subst_from_node(self, node):  # follow the color of expand node
        self.subst = node.subst
        if len(self.mand_vars) > 0:
            for v1, v2 in zip(self.mand_vars, node.mand_vars):
                v1.copy_subst_from_node(v2)

        if len(self.hps) > 0:
            for h1, h2 in zip(self.hps, node.hps):
                h1.copy_subst_from_node(h2)

    def draw_graph(self, output_dir='visualization/', output_format='png', name=''):

        vocab_dict = {}

        def get_graph(g, node, node_idx_str):

            if node.subst:
                g.attr('node', shape='box', style='filled', color='red')
            else:
                g.attr('node', shape='box', style="", color='black')
            label_idx_str = 'label_{}'.format(len(vocab_dict))
            g.node(label_idx_str, node.label)
            vocab_dict[label_idx_str] = node.label
            g.edge(label_idx_str, node_idx_str)

            if len(node.mand_vars) > 0:
                for var in node.mand_vars:
                    if var.subst:
                        g.attr('node', shape='circle', style='filled', color='red')
                    else:
                        g.attr('node', shape='circle', style="", color='black')
                    var_idx_str = 'var_{}'.format(len(vocab_dict))
                    g.node(var_idx_str, var.str)
                    vocab_dict[var_idx_str] = var.str
                    g.edge(var_idx_str, label_idx_str)
                    g = get_graph(g, var, var_idx_str)

            if len(node.hps) > 0:
                for hp in node.hps:
                    if hp.subst:
                        g.attr('node', shape='circle', style='filled', color='red')
                    else:
                        g.attr('node', shape='circle', style="", color='black')
                    hp_idx_str = 'hp_{}'.format(len(vocab_dict))
                    g.node(hp_idx_str, hp.str)
                    vocab_dict[hp_idx_str] = hp.str
                    g.edge(hp_idx_str, label_idx_str)
                    g = get_graph(g, hp, hp_idx_str)

            return g

        G = Digraph(comment='Proof of {}'.format(self.str))
        if self.subst:
            G.attr('node', shape='circle', style='filled', color='red')
        else:
            G.attr('node', shape='circle', style="", color='black')

        G.attr('node', shape='circle')
        self_idx_str = 'res_{}'.format(len(vocab_dict))  # changed this to res since it is not a var node
        G.node(self_idx_str, self.str)
        vocab_dict[self_idx_str] = self.str
        G = get_graph(G, self, self_idx_str)
        if name != '':
            output_name = name
        else:
            output_name = self.name
        G.render('{0}prooftree_{1}'.format(output_dir, output_name), format=output_format, view=False)
        return G

    @staticmethod
    def get_fill_color(subst, color_1='red', color_2='white'):
        subst = round(float(subst), 2)
        if subst == 1:
            res = color_1
        elif subst == 0:
            res = color_2
        else:
            res = '{0};{1}:{2}'.format(color_1, subst, color_2)
        return res

    def draw_graph_2(self, output_dir='visualization/', output_format='png', name=''):

        vocab_dict = {}

        def get_graph(g, node, node_idx_str):

            g.attr('node', shape='box', style='filled', fillcolor=self.get_fill_color(node.subst))
            label_idx_str = 'label_{}'.format(len(vocab_dict))
            g.node(label_idx_str, node.label)
            vocab_dict[label_idx_str] = node.label
            g.edge(label_idx_str, node_idx_str)

            if len(node.mand_vars) > 0:
                for var in node.mand_vars:
                    g.attr('node', shape='circle', style='filled', fillcolor=self.get_fill_color(var.subst))
                    var_idx_str = 'var_{}'.format(len(vocab_dict))
                    g.node(var_idx_str, var.str)
                    vocab_dict[var_idx_str] = var.str
                    g.edge(var_idx_str, label_idx_str)
                    g = get_graph(g, var, var_idx_str)

            if len(node.hps) > 0:
                for hp in node.hps:
                    g.attr('node', shape='circle', style='filled', fillcolor=self.get_fill_color(hp.subst))
                    hp_idx_str = 'hp_{}'.format(len(vocab_dict))
                    g.node(hp_idx_str, hp.str)
                    vocab_dict[hp_idx_str] = hp.str
                    g.edge(hp_idx_str, label_idx_str)
                    g = get_graph(g, hp, hp_idx_str)

            return g

        G = Digraph(comment='Proof of {}'.format(self.str))

        G.attr('node', shape='circle', style='filled', fillcolor=self.get_fill_color(self.subst))

        G.attr('node', shape='circle')
        self_idx_str = 'res_{}'.format(len(vocab_dict))  # changed this to res since it is not a var node
        G.node(self_idx_str, self.str)
        vocab_dict[self_idx_str] = self.str
        G = get_graph(G, self, self_idx_str)
        if name != '':
            output_name = name
        else:
            output_name = self.name
        G.render('{0}prooftree_{1}'.format(output_dir, output_name), format=output_format, view=False)
        return G

    def draw_graph_3(self, output_dir='visualization/', output_format='png', name='', shape='none', fontsize='20',
                     style='rounded,filled', text_label="N: {}\lPROP: {}", fontname='monospace', color1='lightskyblue1',
                     color2='lemonchiffon'):

        vocab_dict = {}

        def get_graph(g, node, node_idx_str):

            for var in node.mand_vars:
                g.attr('node', shape=shape, style=style,
                       fillcolor=self.get_fill_color(var.subst, color_1=color1, color_2=color2))
                label_idx_str = 'res_{}'.format(len(vocab_dict))
                # g.node(label_idx_str, text_label.format(var.label, var.str), fixed_size='shape', fontsize=fontsize, fontname=fontname)
                g.node(label_idx_str, text_label.format(var.label, var.str), fontsize=fontsize, fontname=fontname)
                vocab_dict[label_idx_str] = text_label.format(var.label, var.str)
                g.edge(label_idx_str, node_idx_str, penwidth='2')
                g = get_graph(g, var, label_idx_str)

            for hp in node.hps:
                g.attr('node', shape=shape, style=style,
                       fillcolor=self.get_fill_color(hp.subst, color_1=color1, color_2=color2))
                label_idx_str = 'res_{}'.format(len(vocab_dict))
                g.node(label_idx_str, text_label.format(hp.label, hp.str), fixed_size='shape', fontsize=fontsize,
                       fontname=fontname)
                vocab_dict[label_idx_str] = text_label.format(hp.label, hp.str)
                g.edge(label_idx_str, node_idx_str, penwidth='2')
                g = get_graph(g, hp, label_idx_str)

            return g

        G = Digraph(comment='Proof of {}'.format(self.str))

        G.attr('node', shape=shape, style=style,
               fillcolor=self.get_fill_color(self.subst, color_1=color1, color_2=color2))
        self_idx_str = 'res_{}'.format(len(vocab_dict))
        G.node(self_idx_str, text_label.format(self.label, self.str), fixed_size='shape', fontsize=fontsize,
               fontname=fontname)
        vocab_dict[self_idx_str] = text_label.format(self.label, self.str)
        G = get_graph(G, self, self_idx_str)
        # G.view()
        if name != '':
            output_name = name
        else:
            output_name = self.name
        G.render('{0}prooftree_{1}'.format(output_dir, output_name), format=output_format, view=False)
        return G


class MM:
    def __init__(self, threshold, raw_proof_max_length):
        self.fs = FrameStack()
        self.labels = {}  # contains previous $e that are out of scope, contains results (dvs, f_hyps, e_hyps, stat) from make_assertion
        self.proofs = {}  # contains the proof of a theorem, use summarize proof to get the original proof
        # self.expand_proofs = collections.defaultdict(list)
        self.threshold = threshold
        self.raw_proof_max_length = raw_proof_max_length
        self.current_subproof_count = None  # keep track of number of $p in a proof
        self.subproof_counts = []

    def read(self, toks):  # read metamath actual content
        self.fs.push()
        label = None
        tok = toks.readc()
        while tok not in (None, '$}'):
            if tok == '$c':
                for tok in toks.readstat(): self.fs.add_c(tok)
            elif tok == '$v':
                for tok in toks.readstat(): self.fs.add_v(tok)
            elif tok == '$f':
                stat = toks.readstat()
                if not label: raise MMError('$f must have label')
                if len(stat) != 2: raise MMError('$f must have be length 2')
                vprint(15, label, '$f', stat[0], stat[1], '$.')
                self.fs.add_f(stat[1], stat[0], label)
                self.labels[label] = ('$f', [stat[0], stat[1]])
                label = None
            elif tok == '$a':
                if not label: raise MMError('$a must have label')
                (dvs, f_hyps, e_hyps, stat) = self.fs.make_assertion(toks.readstat())
                self.labels[label] = ('$a', (dvs, f_hyps, e_hyps, stat))
                label = None

            elif tok == '$e':
                if not label: raise MMError('$e must have label')
                stat = toks.readstat()
                self.fs.add_e(stat, label)
                self.labels[label] = ('$e', stat)
                label = None
            elif tok == '$p':
                if not label: raise MMError('$p must have label')
                stat = toks.readstat()
                proof = None
                try:
                    i = stat.index('$=')  # everything between $= and $.
                    proof = stat[i + 1:]  # statement to prove
                    stat = stat[:i]
                except ValueError:
                    raise MMError('$p must contain proof after $=')
                vprint(1, 'verifying', label)

                (dvs, f_hyps, e_hyps, stat) = self.fs.make_assertion(stat)

                self.verify_custom(stat, proof, label)
                self.verify_custom(stat, proof, label, num_expand=1)
                self.labels[label] = ('$p', (dvs, f_hyps, e_hyps, stat))

                label = None
            elif tok == '$d':
                self.fs.add_d(toks.readstat())
            elif tok == '${':  # recursive call because of new frame
                self.read(toks)
            elif tok[0] != '$':  # first get the label
                label = tok
            else:
                print('tok:', tok)
            tok = toks.readc()
        self.fs.pop()

    def apply_subst(self, stat, subst):
        result = []
        for tok in stat:
            if tok in subst:
                result.extend(subst[tok])
            else:
                result.append(tok)
        vprint(20, 'apply_subst', (stat, subst), '=', result)
        return result

    def find_vars(self, stat):  # get all the distinct variables from stat and put them in vars
        vars = []
        for x in stat:
            if not x in vars and self.fs.lookup_v(x): vars.append(x)
        return vars

    def decompress_proof(self, stat, proof):  # decompress proofs into list of labels
        dm, mand_hyp_stmts, hyp_stmts, stat = self.fs.make_assertion(stat)
        # get mandatory and essential hypothesis labels
        mand_hyps = [self.fs.lookup_f(v) for k, v in mand_hyp_stmts]
        hyps = [self.fs.lookup_e(s) for s in hyp_stmts]

        labels = mand_hyps + hyps
        hyp_end = len(labels)
        ep = proof.index(')')
        labels += proof[1:ep]
        compressed_proof = ''.join(proof[ep + 1:])

        vprint(5, 'labels:', labels)
        vprint(5, 'proof:', compressed_proof)

        proof_ints = []
        cur_int = 0

        for ch in compressed_proof:  # Compressed Proof Expansion
            if ch == 'Z':
                proof_ints.append(-1)
            elif 'A' <= ch and ch <= 'T':
                cur_int = (20 * cur_int + ord(ch) - ord('A') + 1)
                proof_ints.append(cur_int - 1)  # convert to zero index
                cur_int = 0
            elif 'U' <= ch and ch <= 'Y':
                cur_int = (5 * cur_int + ord(ch) - ord('U') + 1)
        vprint(5, 'proof_ints:', proof_ints)

        label_end = len(labels)
        decompressed_ints = []
        subproofs = []
        prev_proofs = []
        for pf_int in proof_ints:
            if pf_int == -1:
                subproofs.append(prev_proofs[-1])
            elif 0 <= pf_int and pf_int < hyp_end:
                prev_proofs.append([pf_int])
                decompressed_ints.append(pf_int)
            elif hyp_end <= pf_int and pf_int < label_end:
                decompressed_ints.append(pf_int)

                step = self.labels[labels[pf_int]]
                step_type, step_data = step[0], step[1]
                if step_type in ('$a', '$p'):
                    sd, svars, shyps, sresult = step_data
                    nshyps = len(shyps) + len(svars)
                    if nshyps != 0:
                        new_prevpf = [s for p in prev_proofs[-nshyps:]
                                      for s in p] + [pf_int]
                        prev_proofs = prev_proofs[:-nshyps]
                        vprint(5, 'nshyps:', nshyps)
                    else:
                        new_prevpf = [pf_int]
                    prev_proofs.append(new_prevpf)
                else:
                    prev_proofs.append([pf_int])
            elif label_end <= pf_int:
                pf = subproofs[pf_int - label_end]
                vprint(5, 'expanded subpf:', pf)
                decompressed_ints += pf
                prev_proofs.append(pf)
        vprint(5, 'decompressed ints:', decompressed_ints)

        return [labels[i] for i in decompressed_ints]

    def propagate(self, proof, name):
        stack = []
        for label in proof:
            typ, dat = self.labels[label]
            proof_node = ProofNode(label, typ,
                                   dat)  # keep track of current subproof for $a and $p, basically top of the stack node
            vprint(10, label, ':', self.labels[label])
            if proof_node.type in ('$a', '$p'):
                (distinct, mand_var, hyp, result) = proof_node.data
                vprint(12, proof_node.type)
                npop = len(mand_var) + len(
                    hyp)  # number of arguments needed for the step. Mandatory variables first and then the hypothesis.
                sp = len(stack) - npop
                if sp < 0: raise MMError('stack underflow')
                subst = {}
                for (k, v) in mand_var:  # mandatory variables that need to be substituted
                    entry_node = stack[sp]  # actual content of proof steps
                    entry = entry_node.expr
                    if entry[0] != k:
                        raise MMError(
                            ("stack entry ({0}, {1}) doesn't match " +
                             "mandatory var hyp {2!s}").format(k, v, entry))
                    subst[v] = entry[1:]
                    proof_node.add_mand_vars(entry_node)
                    sp += 1
                vprint(15, 'subst:', subst)
                for x, y in distinct:  # substitute distinct with actual variable
                    vprint(16, 'dist', x, y, subst[x], subst[y])
                    x_vars = self.find_vars(subst[x])
                    y_vars = self.find_vars(subst[y])
                    vprint(16, 'V(x) =', x_vars)
                    vprint(16, 'V(y) =', y_vars)
                    for x, y in itertools.product(x_vars, y_vars):
                        if x == y:  # no need for the look up d
                            raise MMError("disjoint violation: {0}, {1}".format(x, y))
                for h in hyp:  # need to substitute variables in hypothesis of actual proof step with the corresponding one in mandatory variables
                    entry_node = stack[
                        sp]  # entry is the actual proof step, could be a hypothesis in the current proof, h is hypothesis that need to be substituted
                    entry = entry_node.expr
                    subst_h = self.apply_subst(h, subst)
                    if entry != subst_h:  # generally speaking, there can be an error here
                        raise MMError(("stack entry {0!s} doesn't match " +
                                       "hypothesis {1!s}")
                                      .format(entry, subst_h))
                    proof_node.add_hps(entry_node)
                    sp += 1
                n_sp = len(stack) - npop
                del stack[n_sp:]
                result_expr = self.apply_subst(result, subst)
                proof_node.set_expr(result_expr)
                stack.append(proof_node)
            elif proof_node.type in ('$e', '$f'):
                proof_node.set_expr(proof_node.data)
                stack.append(proof_node)
            vprint(12, 'st:', stack)
        stack[0].set_name(name)
        if len(stack) != 1: raise MMError('stack has >1 entry at end')
        assert stack[0].summarize_proof() == proof
        return stack[0]

    def propagate_and_substitute_leaf_hps(self, proof, name):
        stack = []
        for label in proof:
            typ, dat = self.labels[label]
            proof_node = ProofNode(label, typ,
                                   dat)  # keep track of current subproof for $a and $p, basically top of the stack node
            vprint(10, label, ':', self.labels[label])
            if proof_node.type in ('$a', '$p'):
                (distinct, mand_var, hyp, result) = proof_node.data
                vprint(12, proof_node.type)
                npop = len(mand_var) + len(
                    hyp)  # number of arguments needed for the step. Mandatory variables first and then the hypothesis.
                sp = len(stack) - npop
                if sp < 0: raise MMError('stack underflow')
                subst = {}
                for (k, v) in mand_var:  # mandatory variables that need to be substituted
                    entry_node = stack[sp]  # actual content of proof steps
                    entry = entry_node.expr
                    if entry[0] != k:
                        raise MMError(
                            ("stack entry ({0}, {1}) doesn't match " +
                             "mandatory var hyp {2!s}").format(k, v, entry))
                    subst[v] = entry[1:]
                    proof_node.add_mand_vars(entry_node)
                    sp += 1
                vprint(15, 'subst:', subst)
                for x, y in distinct:  # substitute distinct with actual variable
                    vprint(16, 'dist', x, y, subst[x], subst[y])
                    x_vars = self.find_vars(subst[x])
                    y_vars = self.find_vars(subst[y])
                    vprint(16, 'V(x) =', x_vars)
                    vprint(16, 'V(y) =', y_vars)
                    for x, y in itertools.product(x_vars, y_vars):
                        if x == y:  # no need for the look up d
                            raise MMError("disjoint violation: {0}, {1}".format(x, y))
                for h in hyp:  # need to substitute variables in hypothesis of actual proof step with the corresponding one in mandatory variables
                    entry_node = stack[
                        sp]  # entry is the actual proof step, could be a hypothesis in the current proof, h is hypothesis that need to be substituted
                    subst_h = self.apply_subst(h, subst)
                    entry_node.expr = subst_h
                    if entry_node.type == '$e':
                        self.labels[entry_node.label] = ('$e', subst_h)  # add the correct local $e hypothesis
                    proof_node.add_hps(entry_node)
                    sp += 1
                n_sp = len(stack) - npop
                del stack[n_sp:]
                result_expr = self.apply_subst(result, subst)
                proof_node.set_expr(result_expr)
                stack.append(proof_node)
            elif proof_node.type in ('$e', '$f'):
                proof_node.set_expr(proof_node.data)
                stack.append(proof_node)
            vprint(12, 'st:', stack)
        stack[0].set_name(name)
        if len(stack) != 1: raise MMError('stack has >1 entry at end')
        assert stack[0].summarize_proof() == proof
        return stack[0]

    def verify_custom(self, stat, proof, name, mode="error", num_expand=0):
        # if name == '2eu2ex':
        #     print()
        original_num_expand = num_expand
        if num_expand == 1:
            proof_summary = self.proofs[name].summarize_proof()
            if len(proof_summary) > self.raw_proof_max_length:
                return
            if self.threshold != -1:
                times = min(self.current_subproof_count, self.threshold)
            else:
                times = self.current_subproof_count
        else:
            times = 1
        if proof[0] == '(': proof = self.decompress_proof(stat, proof)
        for i in range(times):
            # sometimes a proof contains multiple subproofs, ways determine how many ways we want to expand, currently only works for num_expand=1 case
            num_expand = original_num_expand
            stack = []
            proof_count = 0
            expand = False  # for expanding $p only

            for label in proof:
                typ, dat = self.labels[label]
                proof_node = ProofNode(label, typ,
                                       dat)  # keep track of current subproof for $a and $p, basically top of the stack node
                vprint(10, label, ':', self.labels[label])
                if proof_node.type in ('$a', '$p'):
                    (distinct, mand_var, hyp, result) = proof_node.data

                    vprint(12, proof_node.type)
                    npop = len(mand_var) + len(
                        hyp)  # number of arguments needed for the step. Mandatory variables first and then the hypothesis.
                    sp = len(stack) - npop

                    if sp < 0:
                        if mode == 'error':
                            raise MMError('stack underflow')
                        else:
                            return False, None
                    subst = {}
                    for (k, v) in mand_var:  # mandatory variables that need to be substituted
                        entry_node = stack[sp]  # actual content of proof steps
                        entry = entry_node.expr
                        if entry[0] != k:
                            if mode == "error":
                                raise MMError(
                                    ("stack entry ({0}, {1}) doesn't match " +
                                     "mandatory var hyp {2!s}").format(k, v, entry))
                            else:
                                return False, None
                        subst[v] = entry[1:]
                        proof_node.add_mand_vars(entry_node)
                        sp += 1

                    vprint(15, 'subst:', subst)
                    for x, y in distinct:  # substitute distinct with actual variable
                        vprint(16, 'dist', x, y, subst[x], subst[y])
                        x_vars = self.find_vars(subst[x])
                        y_vars = self.find_vars(subst[y])
                        vprint(16, 'V(x) =', x_vars)
                        vprint(16, 'V(y) =', y_vars)
                        for x, y in itertools.product(x_vars, y_vars):
                            if mode == "error":
                                if name == '' or 'expand' in name:  # in this case, only need to check x == y case
                                    if x == y:
                                        print("disjoint violation")
                                        raise MMError("disjoint violation: {0}, {1}".format(x, y))
                                elif x == y or not self.fs.lookup_d(x, y):  # most likely error due to second condition
                                    print("disjoint violation")
                                    raise MMError("disjoint violation: {0}, {1}".format(x, y))
                            else:
                                return False, None

                    for h in hyp:  # need to substitute variables in hypothesis of actual proof step with the corresponding one in mandatory variables
                        entry_node = stack[
                            sp]  # entry is the actual proof step, could be a hypothesis in the current proof, h is hypothesis that need to be substituted
                        entry = entry_node.expr
                        subst_h = self.apply_subst(h, subst)
                        if entry != subst_h:
                            if mode == "error":
                                raise MMError(("stack entry {0!s} doesn't match " +
                                               "hypothesis {1!s}")
                                              .format(entry, subst_h))
                            else:
                                return False, None
                        proof_node.add_hps(entry_node)
                        sp += 1
                    n_sp = len(stack) - npop

                    del stack[n_sp:]

                    result_expr = self.apply_subst(result, subst)
                    proof_node.set_expr(result_expr)
                    if proof_node.type in '$p':
                        proof_count += 1
                    # remove incomplete and empty proofs that we cannot substitute
                    if num_expand > 0 and proof_node.type in ('$p') and proof_count - 1 == i and len(
                            self.proofs[label].summarize_proof()) <= self.raw_proof_max_length and label not in [
                        'dummylink', 'idi', 'iin1', 'iin3']:
                        # temporary variable will belong to mandatory hypothesis, but will not appear in labels.
                        expand_node = copy.deepcopy(self.proofs[label])
                        expand_node.color_all()
                        expand_subst = {}
                        assert len(mand_var) == len(proof_node.mand_vars)
                        for v, x in zip(mand_var, proof_node.mand_vars):
                            expand_subst[v] = x
                        assert len(hyp) == len(proof_node.hps)
                        for v, x in zip(hyp, proof_node.hps):
                            expand_subst[tuple(v)] = x  # make it hashable
                        reserved_labels = set()
                        for k, v in expand_subst.items():
                            for current_label in v.summarize_proof():  # need to add reserved labels recursively
                                reserved_labels.add(current_label)
                        leaves = expand_node.get_leaves()
                        proof_1 = expand_node.summarize_proof()  # old proof
                        self.avoid_conflict(leaves, reserved_labels, expand_subst)
                        proof_2 = expand_node.summarize_proof()  # new proof
                        if proof_1 != proof_2:
                            expand_node = self.propagate(proof_2, label)
                            expand_node.color_all()  # previous call make expand_node colorless
                            # print(proof_1)
                            # print(proof_2)
                        expand_node.expand_proof(expand_subst)
                        _, proof_node = self.verify_custom(proof_node.expr, expand_node.summarize_proof(),
                                                           "")  # at this point, expand_node and proof_node should give the same summary, the expressions should update themselves in this call
                        proof_node.copy_subst_from_node(expand_node)
                        # proof_node.subst = True  # obsolete from mark_subst
                        num_expand -= 1
                        expanded_label = label
                        expand = True
                    stack.append(proof_node)

                elif proof_node.type in ('$e', '$f'):
                    # add the hypotheses and variable definitions into stack
                    proof_node.set_expr(proof_node.data)
                    stack.append(proof_node)

                vprint(12, 'st:', stack)
            stack[0].set_name(name)
            if len(stack) != 1:
                if mode == "error":
                    raise MMError('stack has >1 entry at end')
                else:
                    return False, None
            if stack[0].expr != stat:
                if mode == "error":
                    raise MMError("assertion proved doesn't match")
                else:
                    return False, stack[0]
            if expand:  # means that was an expanded proof
                _, proof_node = self.verify_custom(stat, stack[0].summarize_proof(),
                                                   "expand_{}_in_{}".format(expanded_label, name))  # get a fresh copy
                # proof_node.mark_subst_old(stack[0])  # color it and maybe draw it, here use the old one to prevent any additional coloring
                proof_node.copy_subst_from_node(stack[0])
                # proof_node.draw_graph()
                # self.proofs[name].draw_graph()
                # self.proofs[expanded_label].draw_graph()
                # self.expand_proofs[self.proofs[name]].append(proof_node)  # reduce memory usage
            elif len(name) > 0:  # preventing dummy verify being added
                assert stack[0].summarize_proof() == proof
                if 'expand' in name and name not in self.proofs:
                    self.proofs[name] = [stack[0]]
                elif 'expand' in name:
                    self.proofs[name].append(stack[0])
                else:
                    self.proofs[name] = stack[0]
            if original_num_expand == 0 and name != '' and 'expand' not in name:
                self.current_subproof_count = proof_count
            if times == 1:
                return True, stack[0]
        # gc.collect()

    def avoid_conflict(self, leaves, reserved_labels, expand_subst):
        replace_dict = {}
        for node in leaves:
            if len(expand_subst) > 0 and tuple(
                    node.expr) not in expand_subst:  # deal with $f not in mand_vars, basically temp variables in the expanding theorem
                if node.type == '$f':
                    if node.label in reserved_labels:
                        if len(node.expr) != 2:
                            assert 0 == 1
                        if node.label not in replace_dict:
                            original_label = node.label
                            new_label = 'sub{0}'.format(len(replace_dict))
                            print('replacing {0} with {1}'.format(node.label, new_label))
                            node.label = new_label
                            node.expr[1] = new_label
                            node.data = node.expr
                            replace_dict[original_label] = [new_label, node.expr]
                            self.labels[new_label] = ('$f', node.expr)
                        else:
                            print(
                                'replacing through dict, {0} with {1}'.format(node.label, replace_dict[node.label][0]))
                            original_label = node.label
                            node.label = replace_dict[node.label][0]
                            node.expr = replace_dict[original_label][1]
                            node.data = node.expr

    def dump(self):
        print(self.labels)


def check_redundancy(proof_node_list, labels):
    summary_list = []
    for proof_node in proof_node_list:
        summary = proof_node.summarize_proof()
        for i in range(len(summary)):
            node = summary[i]
            if labels[node][0] in ['$f', '$e']:
                summary[i] = ''
        summary_list.append(tuple(summary))
    for i in range(len(summary_list) - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            if summary_list[i] == summary_list[j]:
                print('{0} is similar to {1}'.format(proof_node_list[i].name, proof_node_list[j].name))


def count_proofs(proofs):
    # 3229 unique name of expanded proof, 3462 expanded proofs, 1360 proofs, 1352 proofs with subproofs
    normal_proof_count = 0
    expand_proof_count = 0
    for k, v in proofs.items():
        if 'expand' in k:
            expand_proof_count += len(v)
        else:
            normal_proof_count += 1
    print(normal_proof_count, expand_proof_count)


def export_expanded_proof(proofs):
    dataset = []
    word_dict = {}
    for k, v in proofs.items():
        if 'expand' in k:
            for i in range(len(v)):
                name = k + '_variant_{0}'.format(i)
                # res = export_single(v[i])
                res = export_single_new(v[i], word_dict, allow_update=True)
                res.insert(0, name)
                dataset.append(res)
    return dataset, word_dict


def export_proofs_with_fixed_word_dict(proofs, word_dict):
    dataset = []
    for k, v in proofs.items():
        if 'expand' not in k:
            res = export_single_new(v, word_dict, allow_update=False)
            res.insert(0, k)
            dataset.append(res)
    return dataset


def get_expression_indices_update_word_dict(expression, d, allow_update):
    if type(expression) == list:
        expression = ' '.join(expression)
    expression_list = list(expression)
    expression_indices = []
    for char in expression_list:
        if char not in d:
            if allow_update:
                d[char] = len(d)
            else:
                raise NotImplementedError('insufficient vocabulary')
        expression_indices.append(d[char])
    return expression_indices


def get_stats(dataset):
    num_nodes_proof = []  # of length dataset, number of nodes per proof
    num_chars_node_expr = []  # of length num_nodes_proof, number of chars per node (expr)
    num_chars_node_operation = []  # of length num_nodes_proof, number of chars per node (operation)
    is_subst_node = []  # of length num_nodes_proof, mark each node as subst or not
    expanding_theorem_dict = {}
    columns = ['number of chars per node expression', 'number of chars per node operation', 'is node subst']
    for i in range(len(dataset)):
        datapoint = dataset[i]
        expanding_theorem = datapoint[0][datapoint[0].find('expand_') + 7:datapoint[0].find('_in_')]
        if expanding_theorem not in expanding_theorem_dict:
            expanding_theorem_dict[expanding_theorem] = 0
        expanding_theorem_dict[expanding_theorem] += 1
        current_num_chars_node_expr = []
        current_num_chars_node_operation = []
        current_is_subst_node = []
        for node in datapoint[3]:
            current_num_chars_node_expr.append(len(node[0]))
            current_num_chars_node_operation.append(len(node[1]))
            current_is_subst_node.append(node[2])
        num_chars_node_expr.extend(current_num_chars_node_expr)
        num_chars_node_operation.extend(current_num_chars_node_operation)
        is_subst_node.extend(current_is_subst_node)
        num_nodes_proof.append(len(datapoint[3]))
    df = pd.DataFrame(num_nodes_proof, columns=['number of nodes per proof'])
    print(df.describe())
    df = pd.DataFrame(list(expanding_theorem_dict.values()), columns=['expanding theorem histogram'])
    print(df.describe())
    df = pd.DataFrame()
    df[columns[0]] = num_chars_node_expr
    df[columns[1]] = num_chars_node_operation
    df[columns[2]] = is_subst_node
    print(df[columns[0]].describe())
    print(df[columns[1]].describe())
    print(df[columns[2]].describe())
    print('total expanded proofs within criteria :{0}'.format(len(num_nodes_proof)))
    print('total expanded proofs :{0}'.format(len(dataset)))
    expanding_theorem_histogram = list(expanding_theorem_dict.values())
    print('dataset entropy is {0}'.format(entropy(expanding_theorem_histogram, base=2)))
    return num_nodes_proof, num_chars_node_expr, num_chars_node_operation, is_subst_node, expanding_theorem_histogram


def filter_dataset(dataset, proof_max_length, node_string_max_length, max_instance_by_theorem):
    delete_indices = []
    for i in range(len(dataset)):
        datapoint = dataset[i]
        if proof_max_length != -1:
            if len(datapoint[3]) > proof_max_length:
                delete_indices.append(i)
                continue
        for node in datapoint[3]:
            if node_string_max_length != -1:
                if len(node[0]) + len(node[1]) > node_string_max_length:
                    delete_indices.append(i)
                    break
    for i in delete_indices[::-1]:
        del dataset[i]
    # filter by max instances
    if max_instance_by_theorem != -1:
        dataset_indices_by_expanding_theorem = {}
        for i in range(len(dataset)):
            datapoint = dataset[i]
            name = datapoint[0]
            expanding_theorem = name[name.find('expand_') + 7:name.find('_in_')]
            if expanding_theorem not in dataset_indices_by_expanding_theorem:
                dataset_indices_by_expanding_theorem[expanding_theorem] = []
            dataset_indices_by_expanding_theorem[expanding_theorem].append(i)
        remaining_indices = []
        for k, v in dataset_indices_by_expanding_theorem.items():
            if len(v) > max_instance_by_theorem:
                remaining_indices.extend(list(np.random.choice(v, max_instance_by_theorem, replace=False)))
            else:
                remaining_indices.extend(v)
        remaining_indices = sorted(remaining_indices)
        dataset = [dataset[index] for index in remaining_indices]
    return dataset


def export_single_new(proof, word_dict, allow_update):
    # expr is string, nested list
    source = []
    target = []
    nodes_list = []

    def get_graph(node, node_index):
        if len(node.mand_vars) > 0:
            for var in node.mand_vars:
                current_node_list = [get_expression_indices_update_word_dict(var.expr, word_dict, allow_update),
                                     get_expression_indices_update_word_dict(var.label, word_dict, allow_update),
                                     int(var.subst)]
                mand_var_node_index = len(nodes_list)
                source.append(mand_var_node_index)
                target.append(node_index)
                nodes_list.append(current_node_list)
                get_graph(var, mand_var_node_index)
        if len(node.hps) > 0:
            for hp in node.hps:
                current_node_list = [get_expression_indices_update_word_dict(hp.expr, word_dict, allow_update),
                                     get_expression_indices_update_word_dict(hp.label, word_dict, allow_update),
                                     int(hp.subst)]
                hp_node_index = len(nodes_list)
                source.append(hp_node_index)
                target.append(node_index)
                nodes_list.append(current_node_list)
                get_graph(hp, hp_node_index)

    root_node_list = [get_expression_indices_update_word_dict(proof.expr, word_dict, allow_update),
                      get_expression_indices_update_word_dict(proof.label, word_dict, allow_update), int(proof.subst)]
    nodes_list.append(root_node_list)
    get_graph(proof, 0)
    return [source, target, nodes_list]


def export_single(proof):
    # proof is an object of proof node
    source = []
    target = []
    node_attribute_dict = {}

    def get_graph(node, node_index):
        operation_node_dict = {'expr': list(node.label), 'subst': node.subst,
                               'type': 'operation'}  # make expr a list and split it
        # operation_node_dict = {'expr': [node.label], 'subst': node.subst, 'type': 'operation'}  # make expr a list no split
        operation_node_index = len(node_attribute_dict)
        source.append(operation_node_index)
        target.append(node_index)
        node_attribute_dict[len(node_attribute_dict)] = operation_node_dict
        if len(node.mand_vars) > 0:
            for var in node.mand_vars:
                mand_var_node_dict = {'expr': var.expr, 'subst': var.subst, 'type': 'mand_var'}
                mand_var_node_index = len(node_attribute_dict)
                source.append(mand_var_node_index)
                target.append(operation_node_index)
                node_attribute_dict[len(node_attribute_dict)] = mand_var_node_dict
                get_graph(var, mand_var_node_index)
        if len(node.hps) > 0:
            for hp in node.hps:
                hp_node_dict = {'expr': hp.expr, 'subst': hp.subst, 'type': 'hp'}
                hp_node_index = len(node_attribute_dict)
                source.append(hp_node_index)
                target.append(operation_node_index)
                node_attribute_dict[len(node_attribute_dict)] = hp_node_dict
                get_graph(hp, hp_node_index)

    current_node_dict = {'expr': proof.expr, 'subst': proof.subst, 'type': 'res'}
    node_attribute_dict[len(node_attribute_dict)] = current_node_dict
    get_graph(proof, 0)
    return [source, target, node_attribute_dict]


def get_group(dataset):
    group_dict = {}
    group_list = []
    for proof in dataset:
        proof_name = proof[0]
        expanding_theorem = proof_name[proof_name.find('expand_') + 7:proof_name.find('_in')]
        if expanding_theorem not in group_dict.keys():
            group_dict[expanding_theorem] = len(group_dict)
        group_list.append(group_dict[expanding_theorem])
    return group_list, group_dict


def export_proof_summary(proofs):
    res = {}
    for k, v in proofs.items():
        if 'expand' in k:
            res[k] = []
            for proof in v:
                res[k].append(proof.summarize_proof())
        else:
            res[k] = v.summarize_proof()
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="theorem expansion")
    parser.add_argument('-m', dest='main_file', type=str, default='raw_dataset/set.mm', help='files to verify')
    parser.add_argument('-t', dest='threshold', type=int, default=-1, help='max expansion per proof')
    parser.add_argument('-i', dest='load_directory', type=str, default='dataset/set_mm/')
    parser.add_argument('-o', dest='output_directory', type=str,
                        default='dataset/set_mm/')  # for training, validation and test set
    parser.add_argument('-s', dest='split', type=str, default='random')
    parser.add_argument('-rpm', dest='raw_proof_max_length', type=int,
                        default=1000)  # only expands raw proof with max length of 1000 (for both expanding and expanded proofs)
    parser.add_argument('-pm', dest='proof_max_length', type=int, default=1000)
    parser.add_argument('-nm', dest='node_max_length', type=int, default=512)
    parser.add_argument('-mi', dest='max_instance_by_theorem', type=int, default=100)
    parser.add_argument('-vmi', dest='validation_max_instance_by_theorem', type=int, default=10)  # also applies to test
    parser.add_argument('-split_by_names', dest='split_by_names', type=int, default=0)
    np.random.seed(47)
    args = parser.parse_args()
    load_path = args.load_directory
    output_path = args.output_directory
    t1 = time.time()
    if not os.path.exists(load_path):
        os.makedirs(load_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(load_path + 'word_dict.pkl'):
        if not os.path.exists(load_path + args.main_file + '_verified_expanded.pkl'):
            with open(args.main_file, "r") as f:
                print('verifying proofs from {0}'.format(args.main_file))
                mm = MM(args.threshold, args.raw_proof_max_length)
                mm.read(toks(f))
            with open(load_path + args.main_file + '_verified_expanded.pkl', 'wb') as f:
                print('saving verified expanded proofs to {0}'.format(load_path))
                pickle.dump(mm, f)
        else:
            with open(load_path + args.main_file + '_verified_expanded.pkl', 'rb') as f:
                print('loading verified expanded proofs from {0}'.format(load_path))
                mm = pickle.load(f)
        print('total elapsed in expansion {0}'.format(time.time() - t1))
        print('exporting proofs to {0}'.format(load_path))
        dataset, word_dict = export_expanded_proof(mm.proofs)
        original_unexpanded_proofs = export_proofs_with_fixed_word_dict(mm.proofs, word_dict)
        with open(load_path + 'word_dict.pkl', 'wb') as f:
            pickle.dump(word_dict, f)
        with open(load_path + 'whole_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        print('exporting raw proofs to {0}'.format(load_path))
        with open(load_path + 'unexpanded_dataset.pkl', 'wb') as f:
            pickle.dump(original_unexpanded_proofs, f)
        proof_summaries = export_proof_summary(mm.proofs)
        with open(load_path + 'proof_summaries.pkl', 'wb') as f:
            pickle.dump(proof_summaries, f)
        with open(load_path + 'raw_proof_labels.pkl', 'wb') as f:
            pickle.dump(mm.labels, f)
    else:
        print('loading proofs from {0}'.format(load_path))
        with open(load_path + 'word_dict.pkl', 'rb') as f:
            word_dict = pickle.load(f)
        with open(load_path + 'whole_dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
    # save word_dict again to output directory
    with open(output_path + 'word_dict.pkl', 'wb') as f:
        pickle.dump(word_dict, f)
    print('finish loading in {0} seconds'.format(time.time() - t1))
    if args.split_by_names:
        print('overwriting the split mode, now simply loading')
        with open(output_path + 'train_proof_names.pkl', 'rb') as f:
            train_proof_names = pickle.load(f)
        with open(output_path + 'valid_proof_names.pkl', 'rb') as f:
            valid_proof_names = pickle.load(f)
        with open(output_path + 'test_proof_names.pkl', 'rb') as f:
            test_proof_names = pickle.load(f)
        all_proof_names = train_proof_names + valid_proof_names + test_proof_names
        dataset = [proof for proof in dataset if proof[0] in all_proof_names]
        assert len(dataset) == len(all_proof_names)
        dataset_index_dict = {}
        for proof in dataset:
            dataset_index_dict[proof[0]] = len(dataset_index_dict)
    else:
        dataset = filter_dataset(dataset, proof_max_length=args.proof_max_length,
                                 node_string_max_length=args.node_max_length,
                                 max_instance_by_theorem=args.max_instance_by_theorem)
    num_nodes_proof, num_chars_node_expr, num_chars_node_operation, is_subst_node, expanding_theorem_histogram = get_stats(
        dataset)
    print('saving stats for all data')
    np.save(output_path + 'num_nodes_proof.npy', num_nodes_proof)
    np.save(output_path + 'num_chars_node_expr.npy', num_chars_node_expr)
    np.save(output_path + 'num_chars_node_operation.npy', num_chars_node_operation)
    np.save(output_path + 'is_subst_node.npy', is_subst_node)
    np.save(output_path + 'expanding_theorem_histogram.npy', expanding_theorem_histogram)
    if not args.split_by_names:
        if args.split == 'random':
            print('split randomly')
            train_dataset, valid_test_dataset = train_test_split(dataset, train_size=0.8, random_state=47)
            valid_dataset, test_dataset = train_test_split(valid_test_dataset, train_size=0.5, random_state=47)
        elif args.split == 'expanding_theorem':
            print('split by expanding theorem')
            group_list, group_dict = get_group(dataset)
            with open(output_path + 'group_dict.pkl', 'wb') as f:
                pickle.dump(group_dict, f)
            gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=47)
            train_indices, validation_indices = list(gss.split(X=dataset, groups=group_list))[0]
            train_dataset = [dataset[i] for i in train_indices]
            valid_test_dataset = [dataset[i] for i in validation_indices]
            gss = GroupShuffleSplit(n_splits=1, train_size=.5, random_state=47)
            remaining_group_list = [group_list[i] for i in validation_indices]
            validation_indices, test_indices = list(gss.split(X=valid_test_dataset, groups=remaining_group_list))[0]
            valid_dataset = [valid_test_dataset[i] for i in validation_indices]
            test_dataset = [valid_test_dataset[i] for i in test_indices]
        else:
            raise NotImplementedError
        # capping
        valid_dataset = filter_dataset(valid_dataset, proof_max_length=-1, node_string_max_length=-1,
                                       max_instance_by_theorem=args.validation_max_instance_by_theorem)
        test_dataset = filter_dataset(test_dataset, proof_max_length=-1, node_string_max_length=-1,
                                      max_instance_by_theorem=args.validation_max_instance_by_theorem)

        train_proof_names = [e[0] for e in train_dataset]
        with open(output_path + 'train_proof_names.pkl', 'wb') as f:
            pickle.dump(train_proof_names, f)

        valid_proof_names = [e[0] for e in valid_dataset]
        with open(output_path + 'valid_proof_names.pkl', 'wb') as f:
            pickle.dump(valid_proof_names, f)

        test_proof_names = [e[0] for e in test_dataset]
        with open(output_path + 'test_proof_names.pkl', 'wb') as f:
            pickle.dump(test_proof_names, f)
    else:
        train_dataset = [dataset[dataset_index_dict[proof_name]] for proof_name in train_proof_names]
        valid_dataset = [dataset[dataset_index_dict[proof_name]] for proof_name in valid_proof_names]
        test_dataset = [dataset[dataset_index_dict[proof_name]] for proof_name in test_proof_names]
    print('train dataset size: {0}'.format(len(train_dataset)))
    print('valid dataset size: {0}'.format(len(valid_dataset)))
    print('test dataset size: {0}'.format(len(test_dataset)))

    num_nodes_proof, num_chars_node_expr, num_chars_node_operation, is_subst_node, expanding_theorem_histogram = get_stats(
        train_dataset)
    print('saving stats')
    np.save(output_path + 'num_nodes_proof_train.npy', num_nodes_proof)
    np.save(output_path + 'num_chars_node_expr_train.npy', num_chars_node_expr)
    np.save(output_path + 'num_chars_node_operation_train.npy', num_chars_node_operation)
    np.save(output_path + 'is_subst_node_train.npy', is_subst_node)
    np.save(output_path + 'expanding_theorem_histogram_train.npy', expanding_theorem_histogram)

    num_nodes_proof, num_chars_node_expr, num_chars_node_operation, is_subst_node, expanding_theorem_histogram = get_stats(
        valid_dataset)
    print('saving stats')
    np.save(output_path + 'num_nodes_proof_valid.npy', num_nodes_proof)
    np.save(output_path + 'num_chars_node_expr_valid.npy', num_chars_node_expr)
    np.save(output_path + 'num_chars_node_operation_valid.npy', num_chars_node_operation)
    np.save(output_path + 'is_subst_node_valid.npy', is_subst_node)
    np.save(output_path + 'expanding_theorem_histogram_valid.npy', expanding_theorem_histogram)

    num_nodes_proof, num_chars_node_expr, num_chars_node_operation, is_subst_node, expanding_theorem_histogram = get_stats(
        test_dataset)
    print('saving stats')
    np.save(output_path + 'num_nodes_proof_test.npy', num_nodes_proof)
    np.save(output_path + 'num_chars_node_expr_test.npy', num_chars_node_expr)
    np.save(output_path + 'num_chars_node_operation_test.npy', num_chars_node_operation)
    np.save(output_path + 'is_subst_node_test.npy', is_subst_node)
    np.save(output_path + 'expanding_theorem_histogram_test.npy', expanding_theorem_histogram)

    with open(output_path + 'train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(output_path + 'valid_dataset.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)
    with open(output_path + 'test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)
    print('done')
