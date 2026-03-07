import jax
from jax import numpy as jnp

from .default import DefaultMutation
from tensorneat.common import I_INF


class HANEATMutation(DefaultMutation):
    """
    HA-NEAT (Heterogeneous Activation NEAT) mutation operator.

    Extends DefaultMutation with:
    1. Random activation on new node creation (instead of identity/default)
    2. One-node-per-generation activation mutation with speciation protection
    3. Prevention of standard per-node activation mutation
    """

    def __init__(
        self,
        conn_add=0.2,
        conn_delete=0.2,
        node_add=0.1,
        node_delete=0.1,
        activation_mutate_rate=0.1,
        max_conns=100,
    ):
        super().__init__(
            conn_add=conn_add,
            conn_delete=conn_delete,
            node_add=node_add,
            node_delete=node_delete,
        )
        self.activation_mutate_rate = activation_mutate_rate
        self.max_conns = max_conns

    @property
    def num_new_conn_markers(self):
        return 3 + self.max_conns

    def _new_node_attrs(self, state, genome, randkey):
        """New nodes get random activation functions."""
        return genome.node_gene.new_random_attrs(state, randkey)

    def __call__(
        self, state, genome, randkey, nodes, conns, new_node_key, new_conn_key
    ):
        assert new_node_key.shape == ()
        assert len(new_conn_key.shape) == 1 and new_conn_key.shape[0] >= 3

        k1, k2, k3 = jax.random.split(randkey, 3)

        nodes, conns = self.mutate_structure(
            state, genome, k1, nodes, conns, new_node_key, new_conn_key
        )
        nodes, conns = self.mutate_activation(
            state, genome, k2, nodes, conns, new_conn_key
        )
        nodes, conns = self.mutate_values(state, genome, k3, nodes, conns)

        return nodes, conns

    def mutate_activation(self, state, genome, randkey, nodes, conns, new_conn_key):
        """Mutate activation of exactly one random hidden node, with marker reassignment."""
        k1, k2, k3 = jax.random.split(randkey, 3)

        r = jax.random.uniform(k1)

        def do_mutate():
            # Pick one random hidden node
            node_key, node_idx = self.choose_node_key(
                k2,
                nodes,
                genome.input_idx,
                genome.output_idx,
                allow_input_keys=False,
                allow_output_keys=False,
            )

            # Compute activation column index
            n_fixed = len(genome.node_gene.fixed_attrs)
            act_col = n_fixed + genome.node_gene.custom_attrs.index("activation")

            # Pick new random activation
            new_act = jax.random.choice(k3, jnp.array(genome.node_gene.activation_indices))

            # Update the node's activation
            new_nodes = nodes.at[node_idx, act_col].set(
                jnp.where(node_idx != I_INF, new_act, nodes[node_idx, act_col])
            )

            # Reassign historical markers on affected connections
            if "historical_marker" in genome.conn_gene.fixed_attrs:
                marker_col = genome.conn_gene.fixed_attrs.index("historical_marker")
                active_mask = ~jnp.isnan(conns[:, 0])
                affected = active_mask & (
                    (conns[:, 0] == node_key) | (conns[:, 1] == node_key)
                )
                # Use markers from new_conn_key[3:] for reassignment
                cumulative_idx = jnp.cumsum(affected) - 1
                extra_markers = new_conn_key[3:]
                new_markers = extra_markers[cumulative_idx]
                old_markers = conns[:, marker_col]
                updated_markers = jnp.where(affected, new_markers, old_markers)
                new_conns = conns.at[:, marker_col].set(updated_markers)
            else:
                new_conns = conns

            return new_nodes, new_conns

        def no_mutate():
            return nodes, conns

        return jax.lax.cond(r < self.activation_mutate_rate, do_mutate, no_mutate)

    def mutate_values(self, state, genome, randkey, nodes, conns):
        """Override to prevent standard activation mutation (HA-NEAT handles it separately)."""
        # Compute activation column index
        n_fixed = len(genome.node_gene.fixed_attrs)
        act_col = n_fixed + genome.node_gene.custom_attrs.index("activation")

        # Save original activations
        original_acts = nodes[:, act_col]

        # Call parent mutate_values (which vmaps gene.mutate over all nodes)
        new_nodes, new_conns = super().mutate_values(state, genome, randkey, nodes, conns)

        # Restore activations to prevent standard mutation from changing them
        new_nodes = new_nodes.at[:, act_col].set(original_acts)

        return new_nodes, new_conns
