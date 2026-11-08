"""Replay successful parsed memory windows after interruptions, never query answers.

Cache keys bind public dialogue, previous memory context and protocol identity.
No keys, headers or private configuration are inspected or saved.
"""
import hashlib
import json
from pathlib import Path

if __package__:
    from .common import save_json
    from .api_meter import BudgetStop
else:
    from common import save_json
    from api_meter import BudgetStop


def install_memory_window_cache(system, meter, directory, identity):
    from simplemem.core.models.memory_entry import MemoryEntry
    cache = Path(directory)
    cache.mkdir(parents=True, exist_ok=True)
    builder = system.memory_builder
    original = builder._generate_memory_entries
    builder.memory_cache_hits = 0

    def generate(dialogues):
        payload = {'protocol': identity, 'window_size': builder.window_size,
                   'overlap_size': builder.overlap_size,
                   'dialogues': [d.model_dump() for d in dialogues],
                   'previous_entries': [e.model_dump() for e in builder.previous_entries]}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = cache / (key + '.json')
        if path.exists():
            stored = json.loads(path.read_text(encoding='utf-8'))
            digest = hashlib.sha256(json.dumps(stored['entries'], sort_keys=True).encode()).hexdigest()
            if stored['key'] != key or digest != stored['entries_sha256']:
                raise BudgetStop('Memory-window cache validation failed')
            entries = [MemoryEntry.model_validate(row) for row in stored['entries']]
            builder.memory_cache_hits += 1
            meter.write({'id': len(meter.records), 'event': 'memory_cache_replay', 'label': 'memory_build',
                         'cache_key': key, 'source_response_ids': stored['source_response_ids'],
                         'entries': len(entries), 'guard_cny': 0})
            print(f'[Memory cache] Reused {len(entries)} entries; no API request for this window', flush=True)
            return entries
        before = len(meter.records)
        entries = original(dialogues)
        if not entries:
            raise BudgetStop('No valid memory entries for this window; stopping instead of silently skipping it')
        rows = [entry.model_dump() for entry in entries]
        save_json(path, {'key': key, 'entries': rows,
                        'entries_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                        'source_response_ids': [r['id'] for r in meter.records[before:] if r['event'] == 'response']})
        return entries

    builder._generate_memory_entries = generate
