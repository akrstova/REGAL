import os
import sys
import time
import numpy as np
import argparse
import networkx as nx
from flask import Flask, request, jsonify
from flask.views import MethodView
import json as JSON

try:
    import cPickle as pickle
except ImportError:
    import pickle
from scipy.sparse import csr_matrix

import xnetmf
from config import *
from alignments import *

# Init app
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'


class Regal(MethodView):

    def __init__(self):
        self.input_matrix = None  # Edgelist of combined input graph
        self.output = None  # Embeddings path
        self.attributes = None  # File with saved numpy matrix of node attributes, or int of number of attributes to synthetically generate
        self.attrvals = 1  # Number of attribute values. Only used if synthetic attributes are generated
        self.dimensions = 16
        self.k = 3  # Controls of landmarks to sample
        self.untillayer = 2  # Calculation until the layer for xNetMF
        self.alpha = 0.01  # Discount factor for further layers
        self.gammastruc = 1  # Weight on structural similarity
        self.gammaattr = 5  # Weight on attributes similarity
        self.numtop = 3  # Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities.
        self.buckets = 1  # Base of log for degree (node feature) binning
        self.g1_nodes = None
        self.g2_nodes = None
        self.sim_measure = None
        self.global_embeddings = None

    def post(self):
        json = request.json
        self.input_matrix = json['matrix']
        self.output = 'emb/custom.emb'
        self.attributes = json['attributes']
        self.g1_nodes = json['g1_nodes']
        self.g2_nodes = json['g2_nodes']
        self.sim_measure = json['sim_measure']

        # Get true alignments
        true_alignments_fname = json['alignments']  # can be changed if desired
        print("true alignments file: ", true_alignments_fname)
        true_alignments = []
        # if os.path.exists(true_alignments_fname):
        #     with open(true_alignments_fname, "rb") as true_alignments_file:
        #         true_alignments = pickle.load(true_alignments_file)
        #         print(true_alignments)

        # Load in attributes if desired (assumes they are numpy array)
        if self.attributes is not None:
            attributes_file = open(self.attributes)
            self.attributes = np.load(attributes_file)  # load vector of attributes in from file
            print(self.attributes.shape)

        # Learn embeddings and save to output
        print("learning representations...")
        before_rep = time.time()
        self.learn_representations()
        after_rep = time.time()
        print("Learned representations in %f seconds" % (after_rep - before_rep))

        # Score alignments learned from embeddings
        embed = np.load(self.output)
        emb1, emb2 = get_embeddings(embed, self.g1_nodes, self.g2_nodes)
        before_align = time.time()
        if self.numtop == 0:
            self.numtop = None
        # alignment_matrix = get_embedding_similarities(emb1, emb2, num_top=self.numtop)
        # Without kd trees
        alignment_matrix = get_embedding_similarities(emb1, emb2, num_top=None, sim_measure=self.sim_measure)

        # Report scoring and timing
        after_align = time.time()
        total_time = after_align - before_align
        print("Align time: "), total_time
        matched_nodes = {}

        if true_alignments is not None:
            topk_scores = [1, 3]
            for k in topk_scores:
                # score, correct_nodes = score_alignment_matrix(alignment_matrix, topk=k, true_alignments=true_alignments)
                matched_nodes, alignment_score, correct_nodes = score_alignment_matrix(alignment_matrix,
                                                                                       topk=None,
                                                                                       true_alignments=true_alignments)
            print(matched_nodes)
            return JSON.dumps(matched_nodes)

    # Should take in a file with the input graph as edgelist (args.input)
    # Should save representations to args.output
    def learn_representations(self):
        nx_graph = nx.read_edgelist(self.input_matrix, nodetype=int, comments="%")
        print("read in graph")
        adj = nx.adjacency_matrix(nx_graph)  # .todense()
        print("got adj matrix")

        graph = Graph(adj, node_attributes=self.attributes)
        max_layer = self.untillayer
        if self.untillayer == 0:
            max_layer = None
        num_buckets = self.buckets  # BASE OF LOG FOR LOG SCALE
        if num_buckets == 1:
            num_buckets = None
        rep_method = RepMethod(max_layer=max_layer,
                               alpha=self.alpha,
                               k=self.k,
                               num_buckets=num_buckets,
                               normalize=True,
                               gammastruc=self.gammastruc,
                               gammaattr=self.gammaattr)
        rep_method.p = graph.N
        if max_layer is None:
            max_layer = 1000
        print("Learning representations with max layer %d and alpha = %f" % (max_layer, self.alpha))
        representations = xnetmf.get_representations(graph, rep_method)
        pickle.dump(representations, open(self.output, "wb"))
        self.global_embeddings = representations


class Embeddings(MethodView):

    def post(self):
        json = request.json
        regal = Regal()
        regal.input_matrix = json['matrix']
        regal.output = 'emb/custom.emb'
        if json['attributes'] is not None:
            attributes_file = open(json['attributes'])
            regal.attributes = np.load(attributes_file)  # load vector of attributes in from file
        regal.g1_nodes = json['g1_nodes']
        regal.g2_nodes = json['g2_nodes']
        regal.sim_measure = json['sim_measure']
        regal.learn_representations()
        return JSON.dumps(regal.global_embeddings.tolist())


if __name__ == '__main__':
    app.add_url_rule('/regal/', view_func=Regal.as_view('regal'))
    app.add_url_rule('/embeddings/', view_func=Embeddings.as_view('embeddings'), methods=['POST'])
    app.run(port=8000, host='0.0.0.0')
