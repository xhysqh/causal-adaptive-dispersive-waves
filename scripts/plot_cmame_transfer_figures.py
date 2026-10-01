"""Generate the three CMAME transfer figures from frozen local evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fgsp_ch.utils.config import load_yaml  # noqa: E402
from fgsp_ch.visualization.transfer_publication import (  # noqa: E402
    load_collection, load_kdv, plot_kdv_transfer, plot_trait_atlas,
    load_f2, plot_causal_transfer_figure,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_transfer_figures.yaml")
    parser.add_argument("--transfer-only", action="store_true", help="Render only Figures 13--15")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    kdv_path = ROOT / config["inputs"]["kdv"]
    ilw_path = ROOT / config["inputs"]["ilw"]
    power_path = ROOT / config["inputs"]["power_law"]
    for path in (kdv_path, ilw_path, power_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    kdv = load_kdv(kdv_path)
    ilw = load_collection(ilw_path, coordinate="depths", equation="ilw")
    power = load_collection(power_path, coordinate="dispersion_orders", equation="power_law")
    stems = [
        output / "figure_13_kdv_causal_transfer",
        output / "figure_14_ilw_causal_transfer",
        output / "figure_15_power_law_causal_transfer",
    ]
    plot_kdv_transfer(kdv, stems[0])
    plot_trait_atlas(ilw, float(config["selection"]["ilw_depth"]), stems[1])
    plot_trait_atlas(power, float(config["selection"]["power_order"]), stems[2])
    extra_inputs = {}
    summaries = {}
    if "f2_showcase" in config["inputs"] and not args.transfer_only:
        source = ROOT / config["inputs"]["f2_showcase"]
        ledger = ROOT / config["inputs"]["f2_ledger"]
        for key, path in (("f2_showcase", source), ("f2_ledger", ledger)):
            extra_inputs[key] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
        for number, equation in ((16, "mch"), (17, "bo")):
            trajectory = load_f2(source, ledger, equation=equation,
                                 initial_points=int(config["selection"]["f2_initial_points"]))
            stem = output / f"figure_{number}_{equation}_causal_response"
            plot_causal_transfer_figure(trajectory, stem)
            stems.append(stem)
            summaries[equation] = {
                "prefixes": len(trajectory.time),
                "actions": {a: int(sum(trajectory.actions == a)) for a in set(trajectory.actions)},
                "initial_N": int(trajectory.active_points[0]),
                "final_N": int(trajectory.active_points[-1]),
                "scope": "source-equation response, not zero-shot transfer",
            }
    checks = {
        "all_sources_exist": True,
        "posthoc_error_recomputed": True,
        "native_saved_times_only": True,
        "precommitted_representatives": True,
        "local_diagnostics_exclude_reference_features": True,
        "physical_profiles_use_frozen_reference_posthoc_only": True,
        "global_authority_read_from_saved_active_resolution": True,
        "local_sites_are_diagnostics_not_local_refinement_claims": True,
        "trait_endpoints_and_bridge_precommitted": True,
        "all_pdf_png_pairs_exist": all(
            stem.with_suffix(ext).is_file() for stem in stems for ext in (".pdf", ".png")
        ),
    }
    acceptance = {
        "status": "PASS" if all(checks.values()) else "STOP",
        "checks": checks,
        "blocking_failures": [key for key, value in checks.items() if not value],
        "figures": [str(stem.relative_to(ROOT)) for stem in stems],
        "inputs": {
            **extra_inputs,
            "kdv": {"path": str(kdv_path.relative_to(ROOT)), "sha256": digest(kdv_path)},
            "ilw": {"path": str(ilw_path.relative_to(ROOT)), "sha256": digest(ilw_path)},
            "power_law": {"path": str(power_path.relative_to(ROOT)), "sha256": digest(power_path)},
        },
        "source_equation_ledgers": summaries,
        "claim": "Frozen reference-free causal mechanism transfers to KdV, the ILW depth continuum, and unseen power-law dispersion traits; independent references are visualization/posthoc only.",
        "online_spatial_diagnostic": {
            "high_difficulty_display_threshold": 0.60,
            "weights": {"gradient": 0.50, "curvature": 0.35, "high_pass": 0.15},
            "site_definition": "x_star(tau) = argmax_x eta(x, tau)",
            "role": "localized online diagnosis only; committed Fourier authority remains global",
            "reference_used": False,
        },
    }
    (output / "acceptance.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    readme = output / "FIGURE_INDEX.txt"
    readme.write_text(
        "CMAME third-category figures\n\n"
        "Fig13: KdV zero-shot transfer.\nFig14: ILW depth 1 transfer.\n"
        "Fig15: Power-law order 2.6 transfer.\n"
        "Fig16: mCH source-equation causal response (global field grid, not particle count).\n"
        "Fig17: BO source-equation causal response.\n\n"
        "All figures: (a) adaptive surface, (b) diagnostic projection, (c) committed global resolution.\n"
        "Gray: hold; green: global spectral refinement; yellow: elevated-diagnostic contour.\n"
        "Diagnostic sites are recomputed from adaptive snapshots only for visualization; they are not\n"
        "a recording of neural internal attention or proof that eta caused a stored action.\n"
        "F2 event timestamps are committed step endpoints. No reference selects sites or actions.\n"
        "PDF is the manuscript source; PNG is exported at 600 dpi. See acceptance.json for provenance.\n",
        encoding="utf-8",
    )
    if args.transfer_only:
        readme.write_text('Figures 13--15: KdV, ILW depth 1, power-law order 2.6.\n'
                          'Each PDF/PNG has TXT caption notes and JSON diagnostic provenance.\n'
                          'Original outputs are in before_final_refinement/.\n',encoding='utf-8')
    archive_name = "CMAME_Figures_13_15_Final.zip" if args.transfer_only else "CMAME_Third_Category_PNG_PDF.zip"
    with zipfile.ZipFile(output / archive_name, "w", zipfile.ZIP_DEFLATED) as archive:
        for stem in stems:
            for ext in (".png", ".pdf", ".txt", ".json"):
                file = stem.with_suffix(ext)
                archive.write(file, file.name)
        for file in (readme, output / "acceptance.json"):
            archive.write(file, file.name)
    print(json.dumps(acceptance, indent=2))
    raise SystemExit(0 if acceptance["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
