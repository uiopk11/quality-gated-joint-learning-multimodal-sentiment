"""Copy installed libraries and dataset without importing the original environment.

The temporary copies are removed before distribution. No dependency installation.
Uses the already-recorded path inventory to avoid enumerating source directories.
"""
import json
from pathlib import Path
from q3_adapter.export_audit import ROOT, ORIGINAL, copy_one, digest, metadata, save

def main():
    inventory = json.loads((ROOT/'q3_adapter/audit/source_inventory_before.json').read_text(encoding='utf-8'))
    records = []
    source_hashes = {}
    skipped = []
    for i, (rel, old) in enumerate(inventory.items()):
        if old['kind'] != 'file':
            continue
        src = ORIGINAL / rel
        if rel.startswith('.venv/Lib/site-packages/') and not any(part in rel.split('/') for part in ('__pycache__', 'include', 'tests', 'test')):
            relative = rel.removeprefix('.venv/Lib/site-packages/')
            record = copy_one(src, ROOT/'q3_adapter/_runtime/site-packages'/relative,
                              'Temporary validation environment, excluded from final package', False, resume_existing=True)
            records.append(record)
            source_hashes[rel] = record['sha256']
        elif rel == 'data/aligned_50.pkl':
            record = copy_one(src, ROOT/'q3_adapter/_runtime/input/aligned_50.pkl',
                              'Temporary real input for smoke test, excluded from final package', False, resume_existing=True)
            records.append(record)
            source_hashes[rel] = record['sha256']
        else:
            try:
                source_hashes[rel] = digest(src)
            except OSError as error:
                skipped.append({'path':rel, 'reason':str(error)})
                print('Cannot hash unrelated source file:', rel, str(error), flush=True)
        if i % 5000 == 0:
            print('Audited source paths', i, flush=True)
    save('temporary_copy_records.json', records)
    save('source_content_baseline.json', source_hashes)
    save('source_hash_skipped.json', skipped)
    save('source_metadata_before_validation.json', {rel: metadata(ORIGINAL/rel) for rel in inventory})
    print('All source files hashed:',len(source_hashes),'Temporary copies:', len(records), flush=True)

if __name__ == '__main__':
    main()
