import gzip
import json
from pysp.stats import *
from pysp.utils.estimation import optimize, partition_data

if __name__ == '__main__':

    # Load the NIPs data
    fin = open('../../data/nips/all_submissions.json', 'rt', encoding='utf-8')
    data = json.load(fin)
    fin.close()

    # Extract the author sets
    papers   = [([v['id'] for v in u['authors']], [v.lower() for v in u['title']]) for u in data]
    authors  = set([v for u in papers for v in u[0]])
    words    = set([v for u in papers for v in u[1]])

    est1  = BernoulliSetEstimator(pseudo_count=1.0e-8, suff_stat={u:0.5 for u in authors})
    est2  = SequenceEstimator(CategoricalEstimator(pseudo_count=1.0, suff_stat={w:1.0/len(words) for w in words}))
    est3  = CompositeEstimator((est1, est2))
    est   = MixtureEstimator([est3]*10)

    model = optimize(papers, est, max_its=10)

    for comp in model.components:
        print(sorted(comp.dists[0].pmap.items(), key=lambda u: u[1], reverse=True)[:10])
        print(sorted(comp.dists[1].dist.pMap.items(), key=lambda u: u[1], reverse=True)[:10])
