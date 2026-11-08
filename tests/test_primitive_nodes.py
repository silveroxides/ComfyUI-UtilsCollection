import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_primitive_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_primitive_test.nodes.utils_nodes import (
        UC_FromSeedCluster,
        UC_MermaidNote,
        UC_MermaidPreview,
        UC_SeedCluster,
        UC_StaticFloat,
        UC_StaticInt,
    )
finally:
    cli_args.cpu = prior_cpu


def test_static_float_schema_supports_precise_shared_values():
    schema = UC_StaticFloat.define_schema()
    value = schema.inputs[0]

    assert schema.node_id == "UC_StaticFloat"
    assert schema.category == "utils/primitive"
    assert value.default == 1.0
    assert value.step == 0.01
    assert value.min == -sys.float_info.max
    assert value.max == sys.float_info.max
    assert UC_StaticFloat.execute(1.2345).result == (1.2345,)


def test_static_integer_remains_the_numeric_pair_compatibility_node():
    schema = UC_StaticInt.define_schema()

    assert schema.node_id == "UC_StaticInt"
    assert schema.category == "utils/primitive"
    assert UC_StaticInt.execute(7).result == (7,)


def test_seed_cluster_generates_main_and_incremented_seed_list():
    schema = UC_SeedCluster.define_schema()
    seed, increment, count = schema.inputs

    assert schema.node_id == "UC_SeedCluster"
    assert seed.control_after_generate is True
    assert seed.min == 0
    assert seed.max == 0xFFFFFFFFFFFFFFFF
    assert count.default == 8
    assert count.min == 1
    assert len(schema.outputs) == 1
    assert schema.outputs[0].io_type == "INT"
    assert schema.outputs[0].is_output_list is True
    assert schema.outputs[0].display_name == "seeds"
    assert UC_SeedCluster.execute(8, 2, 4).result == (
        [8, 10, 12, 14],
    )
    assert UC_SeedCluster.execute(8, 2).result == (
        [8, 10, 12, 14, 16, 18, 20, 22],
    )


def test_seed_cluster_wraps_at_comfy_seed_limit():
    maximum = 0xFFFFFFFFFFFFFFFF

    assert UC_SeedCluster.execute(maximum, 2).result == (
        [maximum, 1, 3, 5, 7, 9, 11, 13],
    )


def test_from_seed_cluster_unpacks_and_fills_eight_integer_outputs():
    schema = UC_FromSeedCluster.define_schema()

    assert schema.node_id == "UC_FromSeedCluster"
    assert schema.is_deprecated is True
    assert schema.is_input_list is False
    assert [output.display_name for output in schema.outputs] == [f"Seed {index}" for index in range(1, 9)]
    assert len(schema.outputs) == 8
    assert UC_FromSeedCluster.execute([8, 10, 12, 14]).result == (
        8, 10, 12, 14, 14, 14, 14, 14,
    )


def test_mermaid_note_schema_and_execution():
    schema = UC_MermaidNote.define_schema()
    assert schema.node_id == "UC_MermaidNote"
    assert schema.display_name == "Mermaid Note"
    assert schema.category == "utils/documentation"
    assert schema.is_output_node is True
    assert len(schema.inputs) == 1
    assert schema.inputs[0].id == "text"
    assert schema.inputs[0].multiline is True
    assert "mermaid" in schema.inputs[0].default
    assert len(schema.outputs) == 1
    assert schema.outputs[0].id == "text"

    out = UC_MermaidNote.execute("graph TD\nA-->B")
    assert out.result == ("graph TD\nA-->B",)
    assert out.ui == {"mermaid_text": ("graph TD\nA-->B",)}


def test_mermaid_preview_schema_and_execution():
    schema = UC_MermaidPreview.define_schema()
    assert schema.node_id == "UC_MermaidPreview"
    assert schema.display_name == "Preview as Mermaid"
    assert schema.category == "utils/documentation"
    assert schema.is_output_node is True
    assert len(schema.inputs) == 1
    assert schema.inputs[0].id == "source"
    assert len(schema.outputs) == 1
    assert schema.outputs[0].id == "text"

    out_str = UC_MermaidPreview.execute("pie title Pets\n\"Cats\" : 45")
    assert out_str.result == ("pie title Pets\n\"Cats\" : 45",)
    assert out_str.ui == {"mermaid_text": ("pie title Pets\n\"Cats\" : 45",)}

    out_dict = UC_MermaidPreview.execute({"chart": "mindmap"})
    assert "mindmap" in out_dict.result[0]
    assert out_dict.ui == {"mermaid_text": (out_dict.result[0],)}
