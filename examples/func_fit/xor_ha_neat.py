from tensorneat.pipeline import Pipeline
from tensorneat import algorithm, genome, problem
from tensorneat.genome import OriginConn, HANEATMutation
from tensorneat.genome.gene.node import DefaultNode
from tensorneat.common import ACT

"""
Solving XOR-3d with HA-NEAT (Heterogeneous Activation NEAT).

HA-NEAT evolves per-node activation functions with:
- Random activation on new node creation
- One-node-per-generation activation mutation limit
- Historical marker reassignment for speciation protection
"""

max_conns = 50

node_gene = DefaultNode(
    activation_options=[ACT.sigmoid, ACT.relu, ACT.tanh, ACT.sin, ACT.identity],
    activation_replace_rate=0.0,  # disable standard activation mutation
)

ha_mutation = HANEATMutation(
    conn_add=0.2,
    conn_delete=0.2,
    node_add=0.1,
    node_delete=0.1,
    activation_mutate_rate=0.1,
    max_conns=max_conns,
)

algorithm = algorithm.NEAT(
    pop_size=10000,
    species_size=20,
    survival_threshold=0.01,
    genome=genome.DefaultGenome(
        node_gene=node_gene,
        conn_gene=OriginConn(),
        mutation=ha_mutation,
        num_inputs=3,
        num_outputs=1,
        max_nodes=7,
        max_conns=max_conns,
        output_transform=ACT.sigmoid,
    ),
)
problem = problem.XOR3d()

pipeline = Pipeline(
    algorithm,
    problem,
    generation_limit=200,
    fitness_target=-1e-6,
    seed=42,
)
state = pipeline.setup()
state, best = pipeline.auto_run(state)
pipeline.show(state, best)

network = algorithm.genome.network_dict(state, *best)
print(algorithm.genome.repr(state, *best))
