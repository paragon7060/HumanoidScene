"""Per-link reporting behind the aggregate v2 rack and obstacle safety force."""

import torch

from kuavo_isaaclab_scene.rl.debug.contact_probe import describe, link_contacts
from kuavo_isaaclab_scene.rl.multi_box.debug.contact_force import (
    maximum_non_rack_force,
    per_body_forces,
)


def test_per_body_split_matches_the_aggregate_safety_force():
    net = torch.zeros(1, 2, 3)
    # The first body carries a 12 N rack contact plus a 6 N contact elsewhere.
    net[0, 0, 0] = 18.0
    net[0, 1, 1] = 4.0
    rack_first = torch.zeros(1, 1, 1, 3)
    rack_first[0, 0, 0, 0] = 12.0
    rack_second = torch.zeros(1, 1, 1, 3)
    rack, obstacle = per_body_forces(net, (rack_first, rack_second), (0, 1))
    torch.testing.assert_close(rack[0], torch.tensor([12.0, 0.0]))
    torch.testing.assert_close(obstacle[0], torch.tensor([6.0, 4.0]))
    torch.testing.assert_close(
        maximum_non_rack_force(net, (rack_first, rack_second), (0, 1)),
        obstacle.amax(dim=-1))


def test_only_reportable_contacts_are_listed_strongest_first():
    bodies = ("zarm_r4_link", "r_twofinger_base")
    positions = torch.tensor([[1.0, 0.0, 1.5], [1.2, -0.1, 1.3]])
    contacts = link_contacts(
        bodies, torch.tensor([12.0, 0.0]), torch.tensor([0.5, 6.0]), positions, 1.0)
    assert [(contact.body, contact.kind, contact.force_n) for contact in contacts] == [
        ("zarm_r4_link", "rack", 12.0),
        ("r_twofinger_base", "obstacle", 6.0),
    ]
    assert "rack zarm_r4_link 12.0 N at (1.00, 0.00, 1.50) m" in describe(contacts)
    assert describe(()) == "no monitored link contact"
    assert describe(contacts, limit=1).endswith("+1 more")

