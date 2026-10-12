"""Build, verify and inspect fixed masks. Never imports or runs a model."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from .common import (PROJECT, DEFAULT_SOURCE, DEFAULT_OUTPUT, EXPECTED_SOURCE_SHA256,
                     array_fingerprint, canonical_json, code_provenance, json_fingerprint, load_saved_split,
                     load_source, read_json, sha256_file, verify_inventory, write_json)
from .reader import MissingDataset, apply_missing, reconstruction_support


def _check_protection(snapshot):
    records = read_json(Path(snapshot)) if isinstance(snapshot, (str, Path)) else snapshot
    for record in records:
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Preserved file changed: {path}")
    return {"passed": True, "file_count": len(records)}


def _default_protection():
    paths = [DEFAULT_SOURCE, PROJECT / "run_baseline.py"]
    original = PROJECT.parent / "E题数据" / "附件2-数据集特征文件" / "aligned_50.pkl"
    if original.is_file():
        paths.append(original)
    for name in ("outputs", "pretrained", "baseline", "vendor", "configs"):
        paths.extend(path for path in (PROJECT / name).rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and path.suffix != ".pyc")
    return [{"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(set(paths))]


def _check_equal(left, right):
    if len(left["views"]) != len(right["views"]):
        raise ValueError("Reproduction view count differs")
    for one, two in zip(left["views"], right["views"]):
        for key in ("evaluation_family", "condition_id"):
            if one[key] != two[key]:
                raise ValueError(f"Reproduction differs: {key}")
        for key in ("masks", "applicable"):
            if one[key].dtype != two[key].dtype or not np.array_equal(one[key], two[key]):
                raise ValueError(f"Reproduction differs: {one['condition_id']}/{key}")
        if canonical_json(one["metadata"]) != canonical_json(two["metadata"]):
            raise ValueError(f"Reproduction metadata differs: {one['condition_id']}")
    if canonical_json(left["panels"]) != canonical_json(right["panels"]):
        raise ValueError("Reproduction panel cohorts differ")


def _application_checks(fields, source, result, unk_id):
    """Inspect every stored view/sample, including inapplicable panel candidates."""
    checked = 0
    before = {key: array_fingerprint(np.asarray(fields[key])) for key in ("text_bert", "audio", "vision", "regression_labels")}
    for view in result["views"]:
        for i, mask in enumerate(view["masks"]):
            raw_text, raw_audio, raw_vision = (np.asarray(fields[key][i]) for key in ("text_bert", "audio", "vision"))
            item = apply_missing(raw_text, raw_audio, raw_vision, mask, unk_id=unk_id)
            support = reconstruction_support(mask, source["observed"][i], source["content"][i])
            length = int(source["lengths"][i])
            if item["audio_lengths"] != length or item["vision_lengths"] != length:
                raise ValueError("Application changed length")
            if not np.array_equal(item["text"][1:], raw_text[1:]):
                raise ValueError("Attention/segment channels changed")
            if not np.array_equal(item["text"][0, ~mask[0]], raw_text[0, ~mask[0]]) or not np.all(item["text"][0, mask[0]] == unk_id):
                raise ValueError("Text replacement failed")
            for m, (name, original) in enumerate((("audio", raw_audio), ("vision", raw_vision)), 1):
                actual = item[name]
                retained = ~mask[m, 1:]
                if not np.array_equal(actual[:-1][retained], original[1:][retained].astype(np.float32)):
                    raise ValueError("Application changed an unmasked A/V position")
                if np.any(actual[:-1][mask[m, 1:]]) or np.any(actual[-1]):
                    raise ValueError("A/V replacement or terminal padding failed")
                if actual.dtype != np.float32:
                    raise ValueError("A/V dtype mismatch")
            if item["text"].dtype != np.int64:
                raise ValueError("BERT token dtype mismatch")
            for m, name in enumerate(("text", "audio", "vision")):
                expected = np.flatnonzero(mask[m] & source["observed"][i, m]) - 1
                if not np.array_equal(np.flatnonzero(support[name]), expected):
                    raise ValueError("Reconstruction support coordinate mismatch")
                native = source["content"][i] & ~source["observed"][i, m]
                if np.any(support[name][np.flatnonzero(native) - 1]):
                    raise ValueError("Native zero/UNK entered reconstruction support")
            checked += 1
    after = {key: array_fingerprint(np.asarray(fields[key])) for key in before}
    if before != after:
        raise ValueError("Application modified source arrays in memory")
    return {"passed": True, "sample_views_checked": checked,
            "native_unobserved_excluded_from_reconstruction": True,
            "no_source_array_mutation": True, "bert_not_executed": True}


def _write_examples(root, raw, sources, results, tokenizer):
    vocab = Path(tokenizer["vocab_path"]).read_text(encoding="utf-8").splitlines()
    chosen = []
    train = results["train"]["views"][0]
    predicates = [("无新增缺失", lambda row: row["plan"]["modality_set"] == "NONE"),
                  ("短样本退化", lambda row: "short_sequence_no_local_missing" in row["reasons"]),
                  ("原始视觉全零", lambda row: row["modalities"]["V"]["original_all_unobserved"] and row["actual"]["any_missing"]),
                  ("共同加附加缺口", lambda row: row["actual"]["synchrony"] == "shared_plus_extra"),
                  ("多段连续缺失", lambda row: any(value["segment_count"] > 1 and value["max_segment_length"] > 1 for value in row["modalities"].values())),
                  ("人工缺口与原始零行重叠", lambda row: row["modalities"]["V"]["overlap_native_unobserved"] > 0 and not row["modalities"]["V"]["original_all_unobserved"])]
    for title, predicate in predicates:
        candidates = [row["index"] for row in train["metadata"] if predicate(row)]
        if candidates:
            chosen.append((title, "train", train, candidates[0]))
    for family, condition, title in (("paired", "shape_8x1", "严格配对八个分离单点"),
                                      ("exploratory_a3like", "a3like_v1_s0", "探索性同步散点视图")):
        for view in results["valid"]["views"]:
            if (view["evaluation_family"], view["condition_id"]) == (family, condition):
                candidates = [row["index"] for row in view["metadata"] if row["applicable"] and row["actual"]["any_missing"]]
                if family == "paired":
                    candidates = sorted(set(candidates) & set(results["valid"]["panels"]["shape"]["indices"]))
                if candidates:
                    chosen.append((title, "valid", view, candidates[0]))
    lines = ["# 掩码应用样例", "", "按固定规则选取首个符合条件的样本，无模型预测或标签筛选。",
             "这些原始词元仅供构造审计展示，不由缺损预测读取器返回。",
             "正文位置采用0基坐标；M=人工遮蔽，Z=原生零行/UNK代理，R=参与重建支持，点号表示否。", ""]
    for title, split, view, index in chosen:
        source, fields = sources[split], raw[split]
        length = int(source["lengths"][index])
        mask = view["masks"][index]
        item = apply_missing(fields["text_bert"][index], fields["audio"][index], fields["vision"][index], mask, unk_id=tokenizer["unk_id"])
        original_tokens = [vocab[int(value)] for value in fields["text_bert"][index, 0, 1:length + 1]]
        masked_tokens = [vocab[int(value)] for value in item["text"][0, 1:length + 1]]
        lines += [f"## {title}", "", f"{split} / {view['evaluation_family']} / {view['condition_id']} / 原index={index} / ID={source['ids'][index]} / L={length}", "",
                  "原始WordPiece：", "```text", " ".join(original_tokens), "```", "遮蔽后的BERT输入正文：", "```text", " ".join(masked_tokens), "```", "", "```text",
                  "位置十位 " + "".join(str(j // 10) for j in range(length)),
                  "位置个位 " + "".join(str(j % 10) for j in range(length))]
        for m, modality in enumerate(("T", "A", "V")):
            missing = mask[m, 1:length + 1]
            observed = source["observed"][index, m, 1:length + 1]
            lines.extend((f"{modality} 人工   " + "".join("M" if x else "." for x in missing),
                          f"{modality} 原零   " + "".join("." if x else "Z" for x in observed),
                          f"{modality} 监督   " + "".join("R" if x else "." for x in (missing & observed))))
        lines += ["```", "", "实际人工缺失率：" + "，".join(f"{m}={view['metadata'][index]['modalities'][m]['rate']:.4f}" for m in ("T", "A", "V")), ""]
    (root / "样例检查.md").write_text("\n".join(lines), encoding="utf-8")


def _reader_checks(root, source_path, results):
    """Exercise the public Dataset and CPU collation, without model execution."""
    from torch.utils.data import DataLoader
    from unittest.mock import patch

    dataset = MissingDataset(root, source_path, "train")
    expected_keys = {"text", "audio", "vision", "audio_lengths", "vision_lengths", "labels", "id", "index"}
    item = dataset.numpy_item(0)
    if set(item) != expected_keys:
        raise ValueError("Prediction item exposed unexpected fields")
    item["text"][0, 1] = -1
    if dataset.numpy_item(0)["text"][0, 1] == -1:
        raise ValueError("Reader item mutates backing data")
    batch = next(iter(DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0)))
    if tuple(batch["text"].shape) != (3, 3, 50) or tuple(batch["audio"].shape) != (3, 50, 74):
        raise ValueError("CPU collation shape mismatch")
    for index in np.flatnonzero(~dataset._source["observed"][:, 2].any(axis=1)):
        if dataset.supervision(int(index))["reconstruction"]["vision"].any():
            raise ValueError("Native all-zero visual sample has supervision")
    train_count = len(dataset)
    del dataset, batch
    reference = MissingDataset(root, source_path, "train", include_reference=True)
    if "reference" not in reference.numpy_item(0):
        raise ValueError("Explicit train reference missing")
    del reference
    counts = {}
    for family, condition, panel in (("main_mixed", "mixed_v1_s0", None),
                                      ("paired", "shape_8x1", "shape"),
                                      ("exploratory_a3like", "a3like_v1_s0", None)):
        if not any(view["evaluation_family"] == family for view in results["valid"]["views"]):
            continue
        dataset = MissingDataset(root, source_path, "valid", family, condition, panel=panel)
        if set(dataset.numpy_item(0)) != expected_keys:
            raise ValueError("Validation prediction item exposed reference/truth")
        try:
            dataset.supervision(0)
        except ValueError:
            pass
        else:
            raise ValueError("Validation exposed training reconstruction supervision")
        if panel and dataset.indices.tolist() != results["valid"]["panels"][panel]["indices"]:
            raise ValueError("Reader panel cohort mismatch")
        counts[f"{family}/{condition}"] = len(dataset)
        del dataset
    # Invalid reference access must fail before deserializing the source.
    with patch("dataset_construction.reader.read_pickle", side_effect=RuntimeError("must not deserialize")):
        try:
            MissingDataset(root, source_path, "valid", include_reference=True)
        except ValueError:
            pass
        else:
            raise ValueError("Validation reference request was not rejected")
    return {"passed": True, "train_samples": train_count, "valid_view_samples": counts,
            "cpu_dataloader_batch_size": 3, "model_executed": False,
            "prediction_keys": sorted(expected_keys), "training_reference_and_support_isolated": True}


def _write_report(root, manifest, acceptance, results):
    lines = ["# 固定缺失数据构造验收", "", f"总体结果：**{'通过' if acceptance['passed'] else '未通过'}**。", "",
             "本报告仅涉及固定数据构造、读取与掩码验收。未加载模型权重，未运行BERT、训练或预测。", "",
             "| 划分 | 主集样本 | 保存视图数 | 全量应用检查（样本×视图） |", "|---|---:|---:|---:|"]
    for split, result in results.items():
        lines.append(f"| {split} | {manifest['source']['summaries'][split]['sample_count']} | {len(result['views'])} | {acceptance['application'][split]['sample_views_checked']} |")
    lines += ["", "## 关键检查", "", "- 源文件SHA256、原划分、ID顺序、标签与固定来源一致。",
              "- 所有人工掩码只在正文范围；CLS、SEP、padding不遮蔽，长度不变。",
              "- 对全部样本视图检查BERT前UNK替换、A/V置零和左移坐标，原始内存数组不被修改。",
              "- 重建监督支持集为正文∩人工遮蔽∩原始可观测代理；原生整行全零/UNK不进入监督，样本仍保留。",
              "- 配对预算、形状与共同适用交集经检查；不适用候选不进入读取接口的面板样本。",
              "- metadata中的区间、段长、缺失率与最终掩码独立核对。",
              "- 公开MissingDataset接口、训练参考隔离、配对共同交集和CPU DataLoader拼批检查通过。",
              "- 探索性补充视图与主混合/配对视图区分，补充开关不改变主集或配对。",
              f"- 同配置重新生成并逐值对比：{'通过' if acceptance.get('reproduced') else '本次未请求；使用 verify --reproduce 执行'}。",
              f"- 保护文件校验：{acceptance.get('preservation', {}).get('file_count', 0)} 个文件保持原SHA256。", "",
              "## 配对面板共同适用样本", "", "| 划分 | 面板 | 可用 | 排除 |", "|---|---|---:|---:|"]
    for split, result in results.items():
        for name, panel in result["panels"].items():
            lines.append(f"| {split} | {name} | {len(panel['indices'])} | {len(panel.get('excluded', []))} |")
    lines += ["", "每条排除原因和共同样本ID见 panels/*.json；实际率、段长/段数、退化及原始零值重叠见 statistics.json 和 metadata/。",
              "", "完整数组、配置、元数据与代码指纹见 manifest.json；详细断言与重现结果见 acceptance.json。",
              "人工缺失率按掩码计数，不等于新增信息损失率或最终全零率。配对无效候选仍保存以供审计，必须按指定panel读取共同交集。", ""]
    (root / "验收报告.md").write_text("\n".join(lines), encoding="utf-8")


def build(args):
    from .generator import default_recipe, default_exploratory_recipe, generate_split
    from .validation import validate_split, statistics_split

    root, source_path = args.output.resolve(), args.source.resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {root}")
    raw, sources, summaries, tokenizer = load_source(source_path)
    recipe = default_recipe()
    exploratory = None if args.no_exploratory else default_exploratory_recipe()
    if exploratory is not None:
        references = [PROJECT.parent / "data_analysis_attachment3" / "分析报告.md",
                      PROJECT / "dataset_construction" / "audits" / "20260924" / "attachment3" / "README.md"]
        exploratory["observation_sources"] = [{"path": str(path), "sha256": sha256_file(path)} for path in references]
        exploratory["uses_attachment3_pickle"] = False
    protection = read_json(args.preservation_snapshot) if args.preservation_snapshot else _default_protection()
    if protection:
        _check_protection(protection)
    results, checks, stats = {}, {}, {}
    for split, source in sources.items():
        print(f"Generate and validate {split}...", flush=True)
        results[split] = generate_split(split, source, recipe, exploratory)
        checks[split] = validate_split(split, source, results[split], recipe, exploratory)
        stats[split] = statistics_split(split, source, results[split])
    # Only make a version directory after all generation/structural validation
    # succeeds. A write failure leaves an explicit incomplete version (no manifest).
    root.mkdir(parents=True)
    write_json(root / "recipe.json", recipe)
    if exploratory is not None:
        write_json(root / "exploratory_recipe.json", exploratory)
    views_index = []
    for split, source in sources.items():
        (root / "source").mkdir(exist_ok=True)
        np.savez_compressed(root / "source" / f"{split}.npz", **source)
        write_json(root / "panels" / f"{split}.json", results[split]["panels"])
        for view in results[split]["views"]:
            family, condition = view["evaluation_family"], view["condition_id"]
            mask_relative = f"masks/{family}/{split}/{condition}.npz"
            metadata_relative = f"metadata/{family}/{split}/{condition}.jsonl"
            (root / mask_relative).parent.mkdir(parents=True, exist_ok=True)
            (root / metadata_relative).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(root / mask_relative, missing=view["masks"], applicable=view["applicable"])
            with (root / metadata_relative).open("w", encoding="utf-8", newline="\n") as stream:
                for record in view["metadata"]:
                    stream.write(canonical_json(record) + "\n")
            views_index.append({"split": split, "evaluation_family": family, "condition_id": condition,
                                "mask_file": mask_relative, "metadata_file": metadata_relative,
                                "shape": list(view["masks"].shape), "dtype": "bool",
                                "applicable_count": int(view["applicable"].sum()),
                                "mask_array_sha256": array_fingerprint(view["masks"]),
                                "metadata_canonical_sha256": json_fingerprint(view["metadata"])})
    write_json(root / "statistics.json", stats)
    _write_examples(root, raw, sources, results, tokenizer)
    manifest = {"schema_version": 1, "dataset_version": "mix_v1", "status": "constructed",
                "source": {"path": str(source_path), "sha256": EXPECTED_SOURCE_SHA256, "summaries": summaries,
                           "label_fingerprints": {split: array_fingerprint(np.asarray(fields["regression_labels"])) for split, fields in raw.items()}},
                "tokenizer": tokenizer, "modality_order": ["T", "A", "V"],
                "mask_coordinates": "original 50 slots; content j maps to j+1; True means artificial missing",
                "interval_coordinates": "zero-based content [start,end)",
                "reconstruction_support": "content AND artificial_missing AND original_observed_proxy",
                "reconstruction_target_policy": "exclude whole originally all-zero A/V rows and native UNK; keep ordinary zero components of nonzero rows",
                "reader_coordinates": {"text_input": "original 50 slots", "audio_vision_input": "drop CLS and right-pad to 50",
                                       "text_reconstruction_support": "49 slots after CLS", "audio_vision_reconstruction_support": "shifted 50 slots, crop to batch max L when needed"},
                "recipe_sha256": json_fingerprint(recipe), "exploratory_recipe_sha256": json_fingerprint(exploratory),
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
                "code_sha256": code_provenance(), "protected_files_before": protection, "views": views_index}
    manifest["files"] = {path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                         for path in sorted(root.rglob("*")) if path.is_file()}
    write_json(root / "manifest.json", manifest)
    del raw, sources, results
    # Reopen the files through the same verification path used by subsequent runs.
    verify(argparse.Namespace(output=root, source=source_path, reproduce=True))
    print(f"Built and accepted: {root}", flush=True)


def verify(args):
    from .generator import generate_split
    from .validation import validate_split, statistics_split

    root = args.output.resolve()
    manifest = read_json(root / "manifest.json")
    verify_inventory(root, manifest)
    if code_provenance() != manifest["code_sha256"]:
        raise ValueError("Construction/reader code differs from the manifest; use a new data version")
    recipe = read_json(root / "recipe.json")
    exploratory = read_json(root / "exploratory_recipe.json") if (root / "exploratory_recipe.json").exists() else None
    if json_fingerprint(recipe) != manifest["recipe_sha256"] or json_fingerprint(exploratory) != manifest["exploratory_recipe_sha256"]:
        raise ValueError("Recipe fingerprint mismatch")
    source_path = args.source.resolve() if args.source else Path(manifest["source"]["path"])
    raw, sources, summaries, tokenizer = load_source(source_path, manifest["source"]["sha256"])
    if tokenizer != manifest["tokenizer"] or summaries != manifest["source"]["summaries"]:
        raise ValueError("Tokenizer/source summary differs")
    checks, applications, results = {}, {}, {}
    saved_stats = read_json(root / "statistics.json")
    for split, source in sources.items():
        print(f"Verify all saved {split} views...", flush=True)
        with np.load(root / "source" / f"{split}.npz", allow_pickle=False) as frozen:
            for key, value in source.items():
                if not np.array_equal(value, frozen[key]):
                    raise ValueError(f"Frozen source mismatch: {split}/{key}")
        if array_fingerprint(np.asarray(raw[split]["regression_labels"])) != manifest["source"]["label_fingerprints"][split]:
            raise ValueError(f"Label fingerprint mismatch: {split}")
        result = results[split] = load_saved_split(root, manifest, split)
        checks[split] = validate_split(split, source, result, recipe, exploratory)
        if canonical_json(statistics_split(split, source, result)) != canonical_json(saved_stats[split]):
            raise ValueError(f"Statistics mismatch: {split}")
        applications[split] = _application_checks(raw[split], source, result, tokenizer["unk_id"])
        if args.reproduce:
            _check_equal(result, generate_split(split, source, recipe, exploratory))
        if exploratory is not None:
            without = generate_split(split, source, recipe, None)
            baseline_views = {"views": [view for view in result["views"] if view["evaluation_family"] != "exploratory_a3like"], "panels": result["panels"]}
            _check_equal(baseline_views, without)
    if sha256_file(source_path) != manifest["source"]["sha256"]:
        raise ValueError("Source changed during verification")
    preservation = _check_protection(manifest["protected_files_before"])
    manifest["status"] = "accepted"
    write_json(root / "manifest.json", manifest)
    acceptance = {"passed": True, "manifest_fingerprint": json_fingerprint(manifest),
                  "structural": checks, "application": applications,
                  "reproduced": bool(args.reproduce), "exploratory_toggle_independence": True,
                  "artifact_integrity": True, "source_unchanged": True, "preservation": preservation,
                  "scope": "data construction and read interface only; no model loading, BERT, training, prediction or metric evaluation"}
    write_json(root / "acceptance.json", acceptance)
    try:
        acceptance["reader_interface"] = _reader_checks(root, source_path, results)
    except Exception:
        manifest["status"] = "acceptance_failed"
        write_json(root / "manifest.json", manifest)
        acceptance["passed"] = False
        write_json(root / "acceptance.json", acceptance)
        raise
    write_json(root / "acceptance.json", acceptance)
    _write_report(root, manifest, acceptance, results)
    print(json.dumps({"passed": True, "sample_views_checked": sum(value["sample_views_checked"] for value in applications.values()),
                      "reproduced": bool(args.reproduce), "preserved_files": preservation["file_count"],
                      "report": str(root / "验收报告.md")}, ensure_ascii=False), flush=True)


def inspect(args):
    dataset = MissingDataset(args.output, args.source, args.split, args.family, args.condition, panel=args.panel)
    matches = np.flatnonzero(dataset.indices == args.index)
    if len(matches) != 1:
        raise ValueError("Requested original index is not in this view/panel's eligible cohort")
    position = int(matches[0])
    item = dataset.numpy_item(position)
    manifest = read_json(args.output / "manifest.json")
    entry = next(entry for entry in manifest["views"] if (entry["split"], entry["evaluation_family"], entry["condition_id"]) == (args.split, args.family, args.condition))
    lines = (args.output / entry["metadata_file"]).read_text(encoding="utf-8").splitlines()
    info = json.loads(lines[args.index])
    support = reconstruction_support(dataset._missing[args.index], dataset._source["observed"][args.index], dataset._source["content"][args.index])
    output = {"metadata": info, "model_input_shapes": {key: list(value.shape) for key, value in item.items() if isinstance(value, np.ndarray)},
              "bert_input_token_ids": item["text"][0].tolist(), "attention": item["text"][1].tolist(),
              "reconstruction_support_counts": {key: int(value.sum()) for key, value in support.items()},
              "note": "Only corrupted input and audit metadata displayed; no model executed"}
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description="Fixed missing-data construction; no training")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Generate a new immutable version and fully verify")
    build_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build_parser.add_argument("--no-exploratory", action="store_true")
    build_parser.add_argument("--preservation-snapshot", type=Path)
    build_parser.set_defaults(func=build)
    verify_parser = commands.add_parser("verify", help="Read and independently validate saved artifacts")
    verify_parser.add_argument("--source", type=Path)
    verify_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify_parser.add_argument("--reproduce", action="store_true")
    verify_parser.set_defaults(func=verify)
    inspect_parser = commands.add_parser("inspect", help="Inspect one original sample index, without inference")
    inspect_parser.add_argument("--source", type=Path)
    inspect_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    inspect_parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    inspect_parser.add_argument("--family", choices=("main_mixed", "paired", "exploratory_a3like"), default="main_mixed")
    inspect_parser.add_argument("--condition", default="mixed_v1_s0")
    inspect_parser.add_argument("--panel")
    inspect_parser.add_argument("--index", type=int, default=0, help="Original split index, not filtered panel position")
    inspect_parser.set_defaults(func=inspect)
    args = parser.parse_args()
    args.func(args)
