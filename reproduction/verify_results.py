"""Validate recorded public-benchmark evidence without any API/config/key access."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

try:
    from .common import ROOT, load_data, select_questions, upstream_f1, save_json
    from .run_experiment import summarize
except ImportError:
    from common import ROOT, load_data, select_questions, upstream_f1, save_json
    from run_experiment import summarize


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(run_name='locomo_window10', write=False):
    directory = ROOT / 'reproduction/results' / run_name
    plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    rows = [json.loads(s) for s in (directory / 'answers.jsonl').read_text(encoding='utf-8').splitlines() if s]
    data, provenance = load_data()
    questions = select_questions(data, plan['sample_indices'], 2, plan['seed'])
    require(questions == plan['questions'], 'Subset differs from pinned official data')
    require(provenance == plan['dataset'], 'Dataset provenance differs')
    expected = {(q['sample_index'], q['qa_index'], arm, trial)
                for q in questions for arm in plan['arms'] for trial in range(1, plan['trials'] + 1)}
    keys = [(r['sample_index'], r['qa_index'], r['arm'], r['trial']) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, 'Missing, duplicate, or unexpected query rows')
    require(len(rows) == 48 and summary['complete'], 'Expected completed 48-row protocol')
    lookup = {(q['sample_index'], q['qa_index']): q for q in questions}
    for row in rows:
        gold = lookup[row['sample_index'], row['qa_index']]
        require(all(row[k] == gold[k] for k in gold), 'Question/reference differs from official data')
        require(abs(upstream_f1(row['answer_prediction'], gold['answer']) - row['f1']) < 1e-12, 'F1 mismatch')
        require(row['input_tokens'] > 0 and row['api_calls'] > 0 and row['retrieved_count'] > 0, 'Empty pipeline result')
    recalculated = summarize(rows, plan['trials'], len(questions))
    require(recalculated == summary['arms'], 'Stored summary differs from recomputed statistics')
    for name, digest in plan['environment']['source_sha256'].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f'Core source changed: {name}')
    for sample in plan['sample_indices']:
        memory = json.loads((directory / f'memory_{sample}.json').read_text(encoding='utf-8'))
        entries = json.loads((directory / f'memory_{sample}_entries.json').read_text(encoding='utf-8'))
        digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        require(digest == memory['memory_sha256'] and len(entries) == memory['entries'], 'Memory checksum/count mismatch')
        index = json.loads((directory / f'index_{sample}.json').read_text(encoding='utf-8'))
        require(index['memory_rows'] == index['indexed_documents'] == len(entries), 'Incomplete lexical index')
        require(index['policy'] == plan['retrieval_index'], 'Index policy differs')
        entry_ids = {entry['entry_id'] for entry in entries}
        for row in (r for r in rows if r['sample_index'] == sample):
            ids = row['retrieved_entry_ids']
            require(len(ids) == len(set(ids)) == row['retrieved_count'], 'Retrieved memory count/uniqueness differs')
            require(set(ids).issubset(entry_ids), 'Retrieved memory ID is not in the frozen bank')
    # Published evidence contains only this run's query attempts. The original
    # machine retains its complete local ledger separately for budget accounting.
    ledger_path = directory / 'query_api_usage.jsonl'
    if not ledger_path.exists():
        ledger_path = directory.parent / 'api_usage.jsonl'
    ledger = [json.loads(s) for s in ledger_path.read_text(encoding='utf-8').splitlines() if s]
    total_calls = sum(r['api_calls'] for r in rows)
    by_id = {r['id']: r for r in ledger if r['event'] == 'response'}
    legacy_map_path = directory.parent / 'delivery_api_map.json'
    legacy = json.loads(legacy_map_path.read_text(encoding='utf-8')) if legacy_map_path.exists() else []
    legacy_map = {(r['sample_index'], r['qa_index'], r['arm'], r['trial']): r['response_ids'] for r in legacy}
    calls, failed_attempts = [], 0
    mapping, offset = [], 0
    for row in rows:
        key = (row['sample_index'], row['qa_index'], row['arm'], row['trial'])
        ids = row.get('api_response_ids', legacy_map.get(key))
        require(ids is not None and len(ids) == row['api_calls'], 'Missing per-question response IDs')
        require(len(ids) == len(set(ids)), 'Repeated response ID within a question')
        require(all(i in by_id for i in ids), 'Response ID absent from ledger')
        block = [by_id[i] for i in ids]
        calls.extend(block)
        attempts = set(row.get('api_attempt_ids', ids))
        row_errors = sum(r['event'] == 'error' and r['id'] in attempts for r in ledger)
        require(row_errors == row.get('failed_api_attempts', 0), 'Failed-attempt count differs')
        failed_attempts += row_errors
        recorded_cost = sum(r.get('guard_cny', 0) for r in ledger if r['id'] in attempts)
        require(abs(recorded_cost - row['peak_price_cny_upper_estimate']) < 1e-9, 'Per-question cost ledger differs')
        label = f"{row['arm']}/trial_{row['trial']}/sample_{row['sample_index']}/qa_{row['qa_index']}"
        require(all(r['label'] == label for r in block), 'Usage block label mismatch')
        require(sum(r['usage']['prompt_tokens'] for r in block) == row['input_tokens'], 'Input usage mismatch')
        require(sum(r['usage']['completion_tokens'] for r in block) == row['output_tokens'], 'Output usage mismatch')
        require(all(r['finish_reason'] == 'stop' and r['max_tokens'] == 16384 and r['thinking'] == 'disabled'
                    and r['transport_streaming'] for r in block), 'Invalid/mixed API conditions')
        mapping.append({'sample_index':row['sample_index'], 'qa_index':row['qa_index'], 'arm':row['arm'],
                        'trial':row['trial'], 'response_ids':[r['id'] for r in block]})
        offset += len(block)
    output = {'status':'passed', 'api_calls_made_by_validation':0, 'query_rows':len(rows), 'questions':len(questions),
              'trials':plan['trials'], 'balanced_categories':dict(Counter(r['category'] for r in rows)),
              'validated_query_api_responses':total_calls, 'first_response_id':calls[0]['id'],
              'recovered_failed_query_attempts': failed_attempts,
              'last_response_id':calls[-1]['id'], 'checks':['official dataset and gold answers','fixed subset and unique paired rows',
              'F1 and all summary statistics','upstream core source hashes','frozen memory checksum',
              'full lexical index coverage','per-question API labels and token usage','normal completion and common API conditions']}
    if write:
        save_json(directory / 'validation.json', output)
        save_json(directory / 'api_response_map.json', mapping)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='locomo_window10')
    parser.add_argument('--write', action='store_true', help='Write derived validation evidence, never edit raw results')
    args = parser.parse_args()
    print(json.dumps(verify(args.run_name, args.write), ensure_ascii=False, indent=2))
