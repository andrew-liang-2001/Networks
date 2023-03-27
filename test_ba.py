from ba import *


def test_initialise_nodes():
    ba = BarabasiAlbert(10)
    if not (ba.N_0 == 10 == ba.G.num_nodes()):
        raise AssertionError("N_0 must be 10")
    elif not (ba.G.num_edges() == 45):
        raise AssertionError("The number of edges must be 45")
    elif not np.all(ba.node_degrees() == 9):
        raise AssertionError("All nodes must have degree 9")


def test_drive():
    """
    Assert that the length of the attachment nodes is twice the number of edges
    """
    ba = BarabasiAlbert(3)
    for i in range(4, 1000):
        ba.drive(i, 1)
        if not (len(ba._attachment_nodes) == 2 * len(ba.G.edges()) == sum(ba.node_degrees())):
            raise AssertionError("Length of attachment nodes must be twice the edges")
