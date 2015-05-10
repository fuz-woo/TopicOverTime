# -*- coding: utf-8 -*-

import os
import random
import math
from numpy import random as nprand
import copy as cp
import corpus as cputil

corpus = cputil.load_corpus("corpus-formatted.csv")

cc = [([w for w in doc if len(w)>1 and not w.isdigit() and not w.lower().islower()],ul) for doc,ul in corpus]
corpus = [d for d in cc if len(d[0])>43]


class TopicOverTime:
    def __init__(self,corpus,alpha=0.1,beta=0.01,gamma=0.01,C=20,n_iter=300):
        self.corpus = corpus
        self.corpus_user = [[u for t,u in ul] for _,ul in self.corpus]
        self.corpus_timestamp = [[t for t,u in ul] for _,ul in self.corpus]
        self.M = len(self.corpus)
        self.NU = sum(map(len,self.corpus_user))
        
        self.udic = list(set([u for d in self.corpus_user for u in d]))
        self.usize = len(self.udic)
        self.C = C
        self.alpha = alpha
        self.gamma = gamma
        self.n_iter = n_iter
        self.communities = [nprand.randint(0,self.C,size=l) for l in map(len,self.corpus_user)]

        self.Calpha = self.C * self.alpha
        self.ugamma = self.usize * self.gamma


        print "corpus: ",self.M
        print "user count: ",self.usize

        self.n_t = [[0 for c in range(self.C)] for m in range(self.M)]
        self.n_t_sum = [0 for m in range(self.M)]
        self.n_c = [{u:0 for u in self.udic} for c in range(self.C)]
        self.n_c_sum = [0 for c in range(self.C)]
        self.ts = [[] for c in range(self.C)]
        self.ts_sum = [0. for c in range(self.C)]

        for m in range(self.M):
            doc_user = self.corpus_user[m]

            for i in range(len(doc_user)):
                u = self.corpus_user[m][i]
                ts = self.corpus_timestamp[m][i]
                c = self.communities[m][i]
                self.n_t[m][c] += 1
                self.n_c[c][u] += 1
                self.n_t_sum[m] += 1
                self.n_c_sum[c] += 1

        self.theta = [[0 for k in range(self.C)] for m in range(self.M)]
        self.rho = [{u:0 for u in self.udic} for c in range(self.C)]
        self.psi = [(1,1) for c in range(self.C)]
        self.nstats = 0

        self.perplexities = []


    def run(self,n_iter=300):
        self.n_iter = n_iter
        for n in range(self.n_iter):
            print "iteration ",n
            self.gibbs_routine()
            self.update_params()
            pplex = self.perplexity()
            print "perplexity: ",pplex
            self.perplexities.append(pplex)

    def gibbs_routine(self):
        for m in range(self.M):
            du = self.corpus_user[m]

            for ui in range(len(du)):
                c = self.communities[m][ui]
                u = self.corpus_user[m][ui]
                ts = self.corpus_timestamp[m][ui]
                self.n_t[m][c] -= 1
                self.n_c[c][u] -= 1
                self.n_t_sum[m] -= 1
                self.n_c_sum[c] -= 1
                
                self.communities[m][ui] = self.sample_community(m,ui)
                newc = self.communities[m][ui]
                
                self.n_t[m][newc] += 1
                self.n_c[newc][u] += 1            
                self.n_t_sum[m] += 1
                self.n_c_sum[newc] += 1
                self.ts[newc].append(ts)
                self.ts_sum[newc] += ts

    def sample_community(self,t,ui):
        u = self.corpus_user[t][ui]
        ts = self.corpus_timestamp[t][ui]
        
        pc = [self.betap(ts,self.psi[c]) * ( self.n_t[t][c] + self.alpha ) * ( self.n_c[c][u] + self.gamma ) / (self.n_c_sum[c] + self.ugamma) for c in range(self.C)]
                
        _sum = sum(pc)
        pc = [pi/_sum for pi in pc]

        sample = self.choice(pc)
        return sample

    def update_params(self):
        self.theta = [[(self.n_t[m][k] + self.alpha) / (self.n_t_sum[m] + self.Calpha) for k in range(self.C)] for m in range(self.M)]
        self.rho = [{u:(self.n_c[c][u] + self.gamma) / (self.n_c_sum[c] + self.ugamma) for u in self.udic} for c in range(self.C)]

        for c in range(self.C):
            _t = self.ts_sum[c] / self.n_c_sum[c]
            _var = sum([(ts-_t)**2 for ts in self.ts[c]]) / self.n_c_sum[c]
            _m = ( _t * (1 - _t) / _var - 1 )
            _a = _t * _m
            _b = (1-_t) * _m
            if _a+_b > 170:
                _a,_b = 170*_a/(_a+_b),170*_b/(_a+_b)
            self.psi[c] = (_a,_b)
            self.ts[c] = []
            self.ts_sum[c] = 0.
            print c,_t,_var,_a,_b
        self.nstats += 1

    def show_communities(self,num_communities=20,num_words=20):
        communities = [sorted(rho.items(),key=lambda (w,c):c, reverse=True) for rho in self.rho]
        communities = [community[:num_words] for community in communities][:num_communities]
        return communities

    def get_theta(self):
        return self.theta

    def get_rho(self):
        rhos = [sorted(rho.items(),key=lambda (w,c):c, reverse=True) for rho in self.rho]
        rhos = [dict(rho) for rho in rhos]
        return rhos

    def perplexity(self):
        thetas = self.theta
        rhos = self.rho

        pplex = 0.
        for m in range(self.M):
            doc = self.corpus_user[m]
            for n in range(len(doc)):
                u = self.corpus_user[m][n]
                likelihood = 0.
                for c in range(self.C):
                    likelihood += self.rho[c][u]*self.theta[m][c]
                pplex += math.log(likelihood) / self.NU
        return math.exp(0. - pplex)

    def betap(self,x,(a,b)):
        B = 0
        try:
            B = math.gamma(a+b) / ( math.gamma(a) * math.gamma(b) )
        except Exception as e:
            print x,a,b
            raise e
        return B * (x**(a-1)) * ((1-x)**(b-1))

    def choice(self,p):
        r = random.random()
        s = 0.0
        for i in range(len(p)):
            s += p[i]
            if s >= r: return i

lda = TopicOverTime(corpus)


























