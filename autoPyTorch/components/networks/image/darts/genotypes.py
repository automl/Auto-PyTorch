from functools import wraps
from collections import namedtuple
import random
import sys

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    #'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

DARTS = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
                         ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
                         ('sep_conv_3x3', 1), ('skip_connect', 0),
                         ('skip_connect', 0), ('dil_conv_3x3', 2)],
                 normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0),
                                                     ('max_pool_3x3', 1),
                                                     ('skip_connect', 2),
                                                     ('max_pool_3x3', 1),
                                                     ('max_pool_3x3', 0),
                                                     ('skip_connect', 2),
                                                     ('skip_connect', 2),
                                                     ('max_pool_3x3', 1)],
                 reduce_concat=[2, 3, 4, 5])


def generate_genotype(gene_function):
    @wraps(gene_function)
    def wrapper(config=None, steps=4):
        concat = range(2, 6)
        gene_normal, gene_reduce = gene_function(config, steps).values()
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
            )
        return genotype
    return wrapper


@generate_genotype
def get_gene_from_config(config, steps=4):
    gene = {'normal': [], 'reduce': []}

    # node 2
    for cell_type in gene.keys():
        first_edge = (config['edge_{}_0'.format(cell_type)], 0)
        second_edge = (config['edge_{}_1'.format(cell_type)], 1)
        gene[cell_type].append(first_edge)
        gene[cell_type].append(second_edge)

    # nodes 3, 4, 5
    for i, offset in zip(range(3, steps+2), [2, 5, 9]):
        for cell_type in gene.keys():
            input_nodes = config['inputs_node_{}_{}'.format(cell_type, i)].split('_')
            for node in input_nodes:
                edge = (config['edge_{}_{}'.format(cell_type, int(node)+offset)],
                        int(node))
                gene[cell_type].append(edge)
    return gene


@generate_genotype
def random_gene(config=None, steps=4):
    gene = {'normal': [], 'reduce': []}

    n = 1
    for i in range(steps):
        for cell_type in gene.keys():
            first_edge = (random.choice(PRIMITIVES),
                          random.randint(0, n))
            second_edge = (random.choice(PRIMITIVES),
                           random.randint(0, n))

            gene[cell_type].append(first_edge)
            gene[cell_type].append(second_edge)
        n += 1
    return gene


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:\n python {} CONFIGS".format(sys.argv[0]))
        sys.exit(1)

    with open('genotypes.py', 'a') as f:
        _nr_random_genes = sys.argv[1]
        for i in range(int(_nr_random_genes)):
            gene = random_gene()
            f.write('DARTS_%d = %s'%(i, gene))
            f.write('\n')
            print(gene)
