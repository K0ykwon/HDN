from src.utils.experiments import build_ablation_variants


def test_build_ablation_variants_generates_expected_names():
    config = {
        "experiment_name": "mnist_hdn",
        "model": {"structural": {}},
        "ablation": {"variants": ["no_split", "no_prune"]},
    }

    variants = build_ablation_variants(config)

    assert [variant["experiment_name"] for variant in variants] == [
        "mnist_hdn_no_split",
        "mnist_hdn_no_prune",
    ]
    assert variants[0]["model"]["structural"]["split"]["enabled"] is False
    assert variants[1]["model"]["structural"]["prune"]["enabled"] is False
