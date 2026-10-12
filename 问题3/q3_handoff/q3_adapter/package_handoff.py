"""Finalize only this external handoff and create/verify its sibling ZIP.

No source-project filesystem access occurs here. Run after temporary validation
libraries/data have been removed from q3_adapter/_runtime. Refuses existing ZIP.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def file_tree(paths):
    tree = {}
    for path in paths:
        node = tree
        for part in path.split('/'):
            node = node.setdefault(part, {})
    lines = ['q3_handoff/']
    def visit(node, prefix=''):
        children = sorted(node.items(), key=lambda item: (not bool(item[1]), item[0]))
        for i, (name, contents) in enumerate(children):
            last = i == len(children)-1
            lines.append(prefix + ('└── ' if last else '├── ') + name + ('/' if contents else ''))
            visit(contents, prefix + ('    ' if last else '│   '))
    visit(tree)
    return '\n'.join(lines)

def main():
    assert ROOT.name == 'q3_handoff'
    runtime = ROOT/'q3_adapter/_runtime'
    assert not runtime.exists() or not any(p.is_file() for p in runtime.rglob('*')), 'Remove temporary runtime first'
    audit = ROOT/'q3_adapter/audit'
    read = lambda name: json.loads((audit/name).read_text(encoding='utf-8'))
    copies = read('copy_records.json')
    originals = {record['relative_path']: record for record in copies}
    cpu, cuda = read('smoke_test.json'), read('smoke_test_cuda.json')
    integrity = read('source_integrity_final.json')
    assert cpu['passed'] and cuda['passed']
    assert not integrity['content_hash_changes'] and not integrity['original_copy_hash_failures']
    bundle = read('bundle_consistency.json')
    assert all(bundle['checks'].values())
    for record in copies:
        assert sha256(ROOT/record['relative_path']) == record['sha256'], record['relative_path']
    for path in (ROOT/'original_source').rglob('*'):
        if path.is_file():
            assert path.relative_to(ROOT).as_posix() in originals, 'Unexpected file in immutable source'

    initial_paths = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file())
    manifest_path = ROOT/'MANIFEST.json'
    all_paths = sorted(set(initial_paths) | {'MANIFEST.json'})
    table = '''CPU与CUDA均成功通过；真实valid样本 `-UacrmKiTn4$_$7`（index=0，正文26）与 `-qDkUB0GgYY$_$5`（index=1，正文48）。

| 检查 | CPU | CUDA |
|---|---|---|
| 标准predict_batch vs adapter logits最大绝对误差 | 0 | 0 |
| 标准predict_batch vs adapter intensity最大绝对误差 | 0 | 0 |
| extract与hook移除后最大误差 | 0 | 0 |
| 类别完全一致 | 是 | 是 |
| 三模态局部/整体遮蔽，形状/长度/原batch/有限数值 | 通过 | 通过 |
| hook数量，前/后 | 0/0 | 0/0 |
| 连续表示梯度与全部参数冻结 | 通过 | 保持eval，未执行GPU反向 |

设备内以上一致性误差为0，不意味着CPU与CUDA或不同batch大小必然逐位相同。与原样复制的正式valid CSV额外交叉核对，两样本类别均一致；CUDA强度最大差异约2.09e-7、概率最大差异约1.26e-7；CPU强度最大差异约1.78e-5、概率约8.96e-6。未改写CSV或调整任何预测结果。相应记录见`formal_csv_crosscheck.json`。

正式最佳checkpoint为epoch=3、purpose=train、method=M0；外置config/preprocessing/provenance与内嵌信息一致，evaluation记录的checkpoint SHA匹配。CPU与CUDA均在原环境依赖的包内临时副本上完成，依赖版本与导入路径见报告。

完整文件树如下。MANIFEST的`files`逐个记录所有其他包文件；MANIFEST本身无法包含自身SHA，因此单列排除说明，由交付ZIP的SHA覆盖。数据集与临时依赖副本不在ZIP中；临时复制文件的来源与校验记录保留供审计。
'''
    readme = ROOT/'README_Q3_HANDOFF.md'
    text = readme.read_text(encoding='utf-8')
    text = text.replace('<!-- FINAL_VALIDATION -->', table)
    text = text.replace('<!-- FILE_TREE -->', '```text\n'+file_tree(all_paths)+'\n```')
    readme.write_text(text, encoding='utf-8')
    entries = []
    for relative in all_paths:
        if relative == 'MANIFEST.json':
            continue
        path = ROOT/relative
        if relative in originals:
            entry = dict(originals[relative])
            entry['sha256_at_packaging'] = sha256(path)
        else:
            purpose = ('Audit evidence and test results' if relative.startswith('q3_adapter/audit/') else
                       'External Q3 adapter, hooks, masking, tests or export tooling' if relative.startswith('q3_adapter/') else
                       'Receiver documentation / inference requirements / minimal example')
            entry = dict(relative_path=relative, original_absolute_path=None,
                         size_bytes=path.stat().st_size, sha256=sha256(path), purpose=purpose,
                         required_for_runtime=relative in ['q3_adapter/__init__.py','q3_adapter/runtime.py',
                             'q3_adapter/model_adapter.py','q3_adapter/hooks.py','q3_adapter/data.py',
                             'q3_adapter/masking.py','q3_adapter/targets.py','requirements_q3.txt'],
                         original_unmodified_copy=False, generated_this_task=True)
        entries.append(entry)
    manifest = {
        'schema_version': 1, 'formal_run': 'mt_m0_avstd_s1111_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'handoff_absolute_path_at_export': str(ROOT),
        'source_project_absolute_path': integrity['source_root'],
        'class_order': ['negative','neutral','positive'],
        'copied_original_file_count': len(originals), 'generated_file_count_including_manifest': len(all_paths)-len(originals),
        'files': entries,
        'self_record': {'relative_path':'MANIFEST.json','original_absolute_path':None,
                        'original_unmodified_copy':False,'generated_this_task':True,
                        'sha256':None,'size_bytes':None,
                        'reason':'Self-hash and self-size are excluded to avoid recursive metadata. External ZIP SHA-256 covers this file.'},
        'large_files_included': [entry['relative_path'] for entry in entries if entry['size_bytes']>100_000_000],
        'manual_large_file_copy_required': [],
        'dataset_not_distributed': {
            'original_absolute_path':read('temporary_data_copy.json')['original_absolute_path'],
            'size_bytes':read('temporary_data_copy.json')['size_bytes'],
            'sha256':read('temporary_data_copy.json')['sha256'],
            'receiver_target':'q3_adapter/_runtime/input/aligned_50.pkl',
            'reason':'Receiver already owns the original data; temporary test copy removed before packaging.'},
        'bert_dependency': {
            'checkpoint_contains_trained_bert_state':True,
            'base_files_still_required_by_unmodified_constructor':['config.json','model.safetensors','vocab.txt'],
            'target':'original_source/pretrained/bert-base-uncased',
            'all_existing_base_and_tokenizer_files_included':True,
            'special_tokens_map_json_existed_in_original':False,
            'special_token_ids':{'UNK':100,'CLS':101,'SEP':102,'PAD':0}},
        'validation': {
            'cpu_passed':True,'cuda_passed':True,
            'cpu_standard_vs_adapter':cpu['standard_vs_adapter'],
            'cuda_standard_vs_adapter':cuda['standard_vs_adapter'],
            'cpu_standard_vs_extraction':cpu['standard_vs_extraction'],
            'cuda_standard_vs_extraction':cuda['standard_vs_extraction'],
            'samples':cpu['samples'],'environment_cpu':cpu['environment'], 'environment_cuda':cuda['environment'],
            'reports':['q3_adapter/audit/smoke_test.json','q3_adapter/audit/smoke_test_cuda.json',
                       'q3_adapter/audit/bundle_consistency.json','q3_adapter/audit/formal_csv_crosscheck.json']},
        'source_preservation': {
            'explicit_content_writes_renames_deletes_creations_issued_to_original':False,
            'all_delivered_copies_sha256_equal_original':True,
            'source_files_hashed_twice':integrity['source_file_count_hashed_twice'],
            'content_hash_changes':[],
            'unrelated_files_without_content_hash':integrity['source_files_without_content_hash'],
            'directory_access_time_changes':len(integrity['metadata_changes_since_first_snapshot']),
            'time_state_requirement_fully_satisfied':False,
            'limitations':[integrity['limitation'],integrity['path_audit_limit'],integrity['directory_enumeration_note']],
            'report':'q3_adapter/audit/source_integrity_final.json'},
        'temporary_validation_copies_removed':True,
        'zip_verification_policy':'After writing ZIP, each member SHA-256 is compared to the corresponding finalized package file; no extraction outside this package.',
    }
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    zip_path = ROOT.with_suffix('.zip')
    if zip_path.exists():
        raise FileExistsError(zip_path)
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for relative in all_paths:
            archive.write(ROOT/relative, arcname='q3_handoff/'+relative)
    expected_names = {'q3_handoff/'+relative for relative in all_paths}
    with zipfile.ZipFile(zip_path,'r') as archive:
        assert set(archive.namelist()) == expected_names
        for relative in all_paths:
            with archive.open('q3_handoff/'+relative) as stream:
                archived_hash = hashlib.file_digest(stream,'sha256').hexdigest()
            assert archived_hash == sha256(ROOT/relative), relative
    print(json.dumps({'zip_absolute_path':str(zip_path),'size_bytes':zip_path.stat().st_size,
                      'sha256':sha256(zip_path),'file_count':len(all_paths),
                      'original_copies':len(originals),'zip_members_sha256_verified':True}, ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
