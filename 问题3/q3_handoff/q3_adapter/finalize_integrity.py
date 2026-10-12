"""Read-only final source audit; output is confined to this package."""
import json
from pathlib import Path
from q3_adapter.export_audit import ROOT, ORIGINAL, digest, metadata, safe_snapshot, save

def main():
    audit = ROOT/'q3_adapter/audit'
    initial = json.loads((audit/'source_inventory_before.json').read_text(encoding='utf-8'))
    before_validation = json.loads((audit/'source_metadata_before_validation.json').read_text(encoding='utf-8'))
    baseline = json.loads((audit/'source_content_baseline.json').read_text(encoding='utf-8'))
    skipped = json.loads((audit/'source_hash_skipped.json').read_text(encoding='utf-8'))
    # A no-atime directory handle trial still showed timestamp changes on this
    # host. Do not repeat enumeration or restore source timestamps. Inspect only
    # known paths and disclose that unknown additions are not independently ruled out.
    current = {rel: dict(kind=prior['kind'], **metadata(ORIGINAL/rel))
               for rel,prior in initial.items() if (ORIGINAL/rel).exists()}
    metadata_changes = []
    for rel in initial.keys() & current.keys():
        diff = {key: [initial[rel][key], current[rel][key]] for key in initial[rel] if initial[rel][key] != current[rel][key]}
        if diff:
            metadata_changes.append({'path': rel, 'kind': initial[rel]['kind'], 'changes': diff})
    changed_content = []
    for rel, expected in baseline.items():
        if digest(ORIGINAL/rel) != expected:
            changed_content.append(rel)
    copies = json.loads((audit/'copy_records.json').read_text(encoding='utf-8'))
    copy_failures = [record['relative_path'] for record in copies if
                     digest(ROOT/record['relative_path']) != record['sha256'] or
                     digest(Path(record['original_absolute_path'])) != record['sha256']]
    source_roots = [ORIGINAL, *ORIGINAL.parents]
    git_markers = [str(p/'.git') for p in source_roots if (p/'.git').exists()]
    result = {
        'source_root': str(ORIGINAL), 'source_entry_count': len(current),
        'source_file_count_hashed_twice': len(baseline),
        'source_files_without_content_hash': skipped,
        'path_audit_method': 'Final metadata checks of all known initial paths; no final recursive enumeration to avoid additional access-time updates.',
        'path_audit_limit': 'Cannot independently exclude unknown added paths without re-enumeration; this task issued no creation, removal, or rename operations under the original project.',
        'added_paths': sorted(current.keys()-initial.keys()),
        'removed_paths': sorted(initial.keys()-current.keys()),
        'content_hash_changes': changed_content, 'original_copy_hash_failures': copy_failures,
        'metadata_changes_since_first_snapshot': metadata_changes,
        'changes_since_hash_baseline_completed': [rel for rel in before_validation if before_validation[rel] != metadata(ORIGINAL/rel)],
        'safe_directory_enumeration_preserved_times': False,
        'directory_enumeration_note': 'Even the no-atime directory handle trial showed timestamp changes on this host; no further enumeration or timestamp restoration was attempted.',
        'git_commands_executed': False, 'git_markers_at_source_or_ancestors': git_markers,
        'git_status_note': 'No Git commands executed. No repository marker at source or ancestors; no Git status to alter.' if not git_markers else 'No Git commands executed; repository files covered where present in source inventory.',
        'time_state_requirement_fully_satisfied': False,
        'limitation': 'Initial ordinary directory listing updated directory atime on NTFS. A few files were read before the first full snapshot, so task-start access times cannot be established. No timestamps were restored or falsified.',
    }
    save('source_integrity_final.json', result)
    assert not changed_content and not copy_failures
    assert not result['added_paths'] and not result['removed_paths']
    assert all(x['kind']=='directory' and set(x['changes'])=={'atime_ns'} for x in metadata_changes)
    print(json.dumps({'source_file_count_hashed_twice': len(baseline),
                      'content_hash_changes': changed_content,
                      'original_copy_hash_failures': copy_failures,
                      'unhashed_unrelated_files': len(skipped),
                      'time_state_requirement_fully_satisfied': False}, ensure_ascii=False), flush=True)
    print('Directory access-time changes:', len(metadata_changes), flush=True)

if __name__ == '__main__':
    main()
