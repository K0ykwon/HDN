from __future__ import annotations

import copy


def build_ablation_variants(config: dict) -> list[dict]:
    ablation_config = config.get("ablation", {})
    variants = ablation_config.get("variants", [])
    if not variants:
        return []

    variant_configs: list[dict] = []
    for variant_name in variants:
        variant = copy.deepcopy(config)
        variant["experiment_name"] = f"{config['experiment_name']}_{variant_name}"
        structural = variant.setdefault("model", {}).setdefault("structural", {})
        structural.setdefault("birth", {}).setdefault("enabled", True)
        structural.setdefault("split", {}).setdefault("enabled", True)
        structural.setdefault("deepen", {}).setdefault("enabled", True)
        structural.setdefault("prune", {}).setdefault("enabled", True)

        if variant_name == "no_split":
            structural["split"]["enabled"] = False
        elif variant_name == "no_deepen":
            structural["deepen"]["enabled"] = False
        elif variant_name == "no_prune":
            structural["prune"]["enabled"] = False
        elif variant_name == "no_balance":
            structural.setdefault("split", {})["heterogeneity_threshold"] = 10.0
            structural.setdefault("birth", {})["loss_threshold"] = 10.0
        elif variant_name == "random_prune":
            structural.setdefault("prune", {})["usage_threshold"] = -1.0
            variant.setdefault("evaluation", {}).setdefault("random_prune_probe", True)
        else:
            continue

        variant["ablation_variant"] = variant_name
        variant_configs.append(variant)
    return variant_configs
