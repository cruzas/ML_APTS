"""Tests for the pipeline-staging contract shared by the CNN model builders.

Every model that defines ``as_model_dict`` builds a graph of layer nodes and
then calls ``BaseModel.set_stage()`` to assign each node a pipeline stage.
``set_stage`` takes no arguments beyond ``self``: it reads ``self.model_dict``
and ``self.config.num_stages``. The correct sequence is therefore

    self.model_dict = model_dict
    self.set_stage()

``my_cnn`` instead called ``self.set_stage(model_dict, config.num_stages)``,
passing exactly the two values ``set_stage`` already reads off ``self``. That
raises ``TypeError: set_stage() takes 1 positional argument but 3 were given``
the moment ``as_model_dict`` runs. It went unnoticed because ``my_cnn`` is the
only model with an ``as_model_dict`` that is absent from the factory's
``MODEL_MAP``, so nothing ever reached it.

These tests cover the contract for all four CNN builders rather than just the
one that was broken, so the same mistake in a registered model would fail here
too.
"""

import inspect

import pytest

from dd4ml.models.cnn.big_cnn import BigCNN
from dd4ml.models.cnn.medium_cnn import MediumCNN
from dd4ml.models.cnn.my_cnn import MyCNN
from dd4ml.models.cnn.simple_cnn import SimpleCNN

CNN_MODELS = pytest.mark.parametrize(
    "cls",
    [SimpleCNN, MediumCNN, BigCNN, MyCNN],
    ids=["simple_cnn", "medium_cnn", "big_cnn", "my_cnn"],
)


def _build_model_dict(cls, num_stages):
    """Construct the model and return its staged model_dict.

    my_cnn takes the config explicitly; the others read it off self. Both are
    accepted so the contract, not the signature, is what is under test.
    """
    config = cls.get_default_config()
    config.input_channels = 1
    config.output_classes = 10
    config.num_stages = num_stages

    model = cls(config)
    takes_config = len(inspect.signature(cls.as_model_dict).parameters) > 1
    return model.as_model_dict(config) if takes_config else model.as_model_dict()


@CNN_MODELS
def test_as_model_dict_assigns_a_stage_to_every_node(cls):
    model_dict = _build_model_dict(cls, num_stages=1)

    assert model_dict, "as_model_dict returned an empty graph"
    missing = [name for name, node in model_dict.items() if "stage" not in node]
    assert not missing, f"nodes with no stage assigned: {missing}"


@CNN_MODELS
@pytest.mark.parametrize("num_stages", [1, 2, 4])
def test_stages_span_exactly_the_requested_range(cls, num_stages):
    """set_stage must use every requested stage and invent none beyond them."""
    model_dict = _build_model_dict(cls, num_stages)
    stages = sorted({node["stage"] for node in model_dict.values()})

    assert stages == list(range(num_stages)), (
        f"{cls.__name__} with num_stages={num_stages} produced stages {stages}"
    )


@CNN_MODELS
def test_stages_are_monotonic_in_graph_order(cls):
    """Nodes are emitted in topological order, so their stages must not go
    backwards -- a later layer cannot live on an earlier pipeline stage."""
    model_dict = _build_model_dict(cls, num_stages=2)
    stages = [node["stage"] for node in model_dict.values()]

    assert stages == sorted(stages), f"{cls.__name__} stage order is not monotonic"


def test_set_stage_takes_no_arguments_beyond_self():
    """Pins the signature the builders above rely on. Passing model_dict and
    num_stages positionally -- as my_cnn used to -- raises TypeError."""
    from dd4ml.models.base_model import BaseModel

    assert list(inspect.signature(BaseModel.set_stage).parameters) == ["self"]

    model = MyCNN.__new__(MyCNN)
    with pytest.raises(TypeError):
        BaseModel.set_stage(model, {}, 1)
