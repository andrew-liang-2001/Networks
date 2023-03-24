import pytest
from ba import *


def test_initialise_nodes():
    ba = BarabasiAlbert(10, graph_type="complete")
    assert ba.N_0 == 10 == ba.G.num_nodes()


def test_initialise_edges():
    ba = BarabasiAlbert(10, graph_type="complete")
    assert ba.G.num_edges() == 45


def test_attachment_nodes():
    ba = BarabasiAlbert(3, graph_type="complete")
    assert len(ba._attachment_nodes) == 6 and ba._attachment_nodes.count(0) == 2


def test_drive():
    ba = BarabasiAlbert(5)
    ba.drive(20, 2)
    assert ba.N_0 == 20 == ba.G.num_nodes()
