"""Small, metered reproduction using the author's original text-memory pipeline.

Default is a dry run: no key loading, model downloads, or API calls.
Execute examples (from repository root):
  .venv/Scripts/python.exe reproduction/run_experiment.py --mode smoke --execute
  .venv/Scripts/python.exe reproduction/run_experiment.py --mode experiment --execute
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

if __package__:
    from .common import ROOT, load_data, save_json, select_questions, upstream_f1
    from .api_meter import Meter, BudgetStop
else:
    from common import ROOT, load_data, save_json, select_questions, upstream_f1
    from api_meter import Meter, BudgetStop

sys.path.insert(0, str(ROOT))

# The single treatment variable: semantic memories retrieved for each search.
# Baseline reproduces the author's default; the modification changes 25 to 10.
EXPERIMENT_VARIANTS = {'baseline_k25': 25, 'modified_k10': 10}


class RedactingStream:
    def __init__(self, stream, secret):
        self.stream, self.secret = stream, secret

    def write(self, value):
        return self.stream.write(value.replace(self.secret, '[REDACTED]') if self.secret else value)

    def flush(self):
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class TeeStream:
    def __init__(self, stream, logfile):
        self.stream, self.logfile = stream, logfile

    def write(self, value):
        result = self.stream.write(value)
        self.logfile.write(value)
        self.logfile.flush()
        return result

    def flush(self):
        self.stream.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def metadata():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = 'unavailable'
    versions = {}
    for name in ['openai', 'torch', 'transformers', 'sentence-transformers', 'numpy', 'lancedb', 'tantivy']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = ['main.py', 'simplemem/core/hybrid_retriever.py', 'simplemem/core/memory_builder.py',
               'simplemem/core/answer_generator.py', 'test_locomo10.py']
    return {'commit': commit, 'python': sys.version, 'dependencies': versions,
            'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}}


def convert_dialogues(sample):
    """Same order/date/caption conversion as the pinned author's evaluator."""
    from simplemem.core.models.memory_entry import Dialogue
    conversation = sample['conversation']
    sessions = sorted([key for key, value in conversation.items()
                       if key.startswith('session_') and isinstance(value, list)],
                      key=lambda value: int(value.split('_')[1]))
    dialogues = []
    for session in sessions:
        timestamp = conversation.get(session + '_date_time')
        if not timestamp:
            continue
        for turn in conversation[session]:
            text = turn.get('text', '')
            if 'img_url' in turn and 'blip_caption' in turn:
                text = f"[Image: {turn['blip_caption']}]" + (f' {text}' if text else '')
            dialogues.append(Dialogue(dialogue_id=len(dialogues) + 1, speaker=turn['speaker'],
                                      content=text, timestamp=timestamp))
    return dialogues


def initialize(bank_path, table_name, meter):
    from main import SimpleMemSystem
    system = SimpleMemSystem(db_path=str(bank_path), table_name=table_name, clear_db=False,
                             use_streaming=False, enable_thinking=False,
                             enable_planning=True, enable_reflection=True,
                             max_reflection_rounds=2, enable_parallel_processing=False,
                             enable_parallel_retrieval=False)
    # Intercept all LLM calls while leaving prompts, planner, retriever, and answer
    # generation implemented by upstream code. No secret or prompt enters the ledger.
    meter.wrap(system.llm_client)
    return system


def ensure_memories(system, dialogues, marker, identity, meter):
    if marker.exists():
        previous = json.loads(marker.read_text(encoding='utf-8'))
        if previous['identity'] != identity:
            raise ValueError('Memory cache identity differs; use another run name')
        memories = system.get_all_memories()
        if not memories:
            raise ValueError('Memory cache marker exists but bank is empty')
        return previous
    if system.get_all_memories():
        raise ValueError('Unfinished memory bank detected; use a new run name')
    before = meter.reserved_cny
    meter.label = 'memory_build'
    started = time.perf_counter()
    system.add_dialogues(dialogues)
    system.finalize()
    memories = system.get_all_memories()
    if not memories:
        raise ValueError('Memory construction produced no entries')
    memory_rows = [memory.model_dump() for memory in memories]
    save_json(marker.with_name(marker.stem + '_entries.json'), memory_rows)
    summary = {'identity': identity, 'dialogues': len(dialogues), 'entries': len(memories),
               'actual_window_size': system.memory_builder.window_size,
               'actual_overlap_size': system.memory_builder.overlap_size,
               'memory_cache_replays': getattr(system.memory_builder, 'memory_cache_hits', 0),
               'seconds': time.perf_counter() - started,
               'peak_price_cny_upper_estimate': meter.reserved_cny - before,
               'memory_sha256': hashlib.sha256(json.dumps(memory_rows, sort_keys=True).encode()).hexdigest()}
    save_json(marker, summary)
    return summary


def score_question(system, question, arm, trial, meter):
    meter.label = f"{arm}/trial_{trial}/sample_{question['sample_index']}/qa_{question['qa_index']}"
    before_cost, before_calls = meter.reserved_cny, len(meter.records)
    started = time.perf_counter()
    retrieval_started = time.perf_counter()
    contexts = system.hybrid_retriever.retrieve(question['question'])
    retrieval_seconds = time.perf_counter() - retrieval_started
    answer = system.answer_generator.generate_answer(question['question'], contexts)
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('Answer generation produced an empty answer')
    calls = [record for record in meter.records[before_calls:] if record['event'] == 'response']
    return {**question, 'arm': arm, 'trial': trial, 'answer_prediction': answer,
            'f1': upstream_f1(answer, question['answer']), 'seconds': time.perf_counter() - started,
            'retrieval_seconds': retrieval_seconds, 'retrieved_count': len(contexts),
            'retrieved_entry_ids': [entry.entry_id for entry in contexts],
            'peak_price_cny_upper_estimate': meter.reserved_cny - before_cost,
            'api_calls': len(calls),
            'api_response_ids': [record['id'] for record in calls],
            'api_attempt_ids': [record['id'] for record in meter.records[before_calls:] if record['event'] == 'reservation'],
            'failed_api_attempts': sum(record['event'] == 'error' for record in meter.records[before_calls:]),
            'input_tokens': sum(record['usage']['prompt_tokens'] for record in calls),
            'output_tokens': sum(record['usage']['completion_tokens'] for record in calls)}


def refresh_full_text_index(system, destination):
    """Freeze a complete lexical index before either comparison arm runs."""
    import tantivy
    table = system.vector_store.table
    table.create_fts_index('lossless_restatement', use_tantivy=True,
                           tokenizer_name='en_stem', replace=True)
    path, _, exists = table._get_fts_index_path()
    if not exists:
        raise ValueError('Full-text index was not created')
    count = tantivy.Index.open(path).searcher().num_docs
    evidence = {'policy': 'rebuild_after_memory_before_queries',
                'memory_rows': table.count_rows(), 'indexed_documents': count}
    if count != evidence['memory_rows']:
        raise ValueError(f'Full-text index is incomplete: {evidence}')
    save_json(destination, evidence)
    print(f'[Index validation] All {count} memories indexed for both arms', flush=True)


def summarize(rows, trials, questions_per_trial):
    groups = defaultdict(list)
    for row in rows:
        groups[row['arm']].append(row)
    summary = {}
    for arm, values in groups.items():
        trial_scores = [statistics.mean(row['f1'] for row in values if row['trial'] == trial)
                        for trial in range(1, trials + 1) if any(row['trial'] == trial for row in values)]
        per_category = {str(category): statistics.mean(row['f1'] for row in values if row['category'] == category)
                        for category in (1, 2, 3, 4) if any(row['category'] == category for row in values)}
        summary[arm] = {'completed_questions': len(values), 'expected_questions': questions_per_trial * trials,
                        'trial_f1': trial_scores, 'mean_f1': statistics.mean(trial_scores),
                        'variance_f1': statistics.variance(trial_scores) if len(trial_scores) > 1 else None,
                        'std_f1': statistics.stdev(trial_scores) if len(trial_scores) > 1 else None,
                        'category_f1': per_category,
                        'category_macro_f1': statistics.mean(per_category.values()),
                        'mean_seconds': statistics.mean(row['seconds'] for row in values),
                        'mean_retrieved_count': statistics.mean(row['retrieved_count'] for row in values),
                        'mean_input_tokens': statistics.mean(row['input_tokens'] for row in values),
                        'mean_output_tokens': statistics.mean(row['output_tokens'] for row in values),
                        'query_api_peak_cny_upper_estimate': sum(row['peak_price_cny_upper_estimate'] for row in values)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['dry-run', 'import-check', 'smoke', 'experiment'], default='dry-run')
    parser.add_argument('--execute', action='store_true', help='Explicitly allow API use')
    parser.add_argument('--samples', type=int, nargs='+', default=[0])
    parser.add_argument('--per-category', type=int, default=2)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--seed', type=int, default=5660, help='Subset/order seed; not an API decoding seed')
    parser.add_argument('--budget-cny', type=float, default=1.0, help='Cumulative local ledger guard, at peak prices')
    parser.add_argument('--run-name', default='locomo_pilot')
    parser.add_argument('--max-output', type=int, default=8192)
    parser.add_argument('--memory-window-size', type=int, default=40)
    parser.add_argument('--cache-memory-windows', action='store_true')
    parser.add_argument('--api-timeout', type=float, default=90)
    parser.add_argument('--console-log', action='store_true', help='Save already-redacted console output in this run directory')
    args = parser.parse_args()
    if not 0 < args.budget_cny <= 30:
        parser.error('Budget must be greater than zero and at most the authorized CNY 30')
    if args.per_category < 1 or args.trials < 1 or not 1 <= args.max_output <= 16384:
        parser.error('Invalid subset, trial, or output size')
    if not args.run_name.replace('_', '').replace('-', '').isalnum():
        parser.error('Run name may contain only letters, digits, hyphens, and underscores')
    if args.memory_window_size < 3:
        parser.error('Memory window size must exceed the default overlap of two')
    if not 30 <= args.api_timeout <= 300:
        parser.error('API timeout must be between 30 and 300 seconds')
    os.environ['WINDOW_SIZE'] = str(args.memory_window_size)
    print(f'[Run configuration] mode={args.mode}; run_name={args.run_name}; '
          f'max_output={args.max_output}; transport_streaming=True', flush=True)
    data, provenance = load_data()
    questions = select_questions(data, args.samples, args.per_category, args.seed)
    plan = {'mode': args.mode, 'sample_indices': args.samples, 'questions': questions,
            'seed': args.seed, 'trials': args.trials, 'arms': EXPERIMENT_VARIANTS,
            'dataset': provenance, 'environment': metadata(),
            'comparability': 'DeepSeek model and a fixed subset differ from the paper. This is a partial reproduction.',
            'retrieval_index': 'rebuild_after_memory_before_queries',
            'memory_window_replay_enabled': args.cache_memory_windows,
            'common_conditions': {'agent_interface_streaming': False, 'thinking': False, 'max_output_tokens': args.max_output,
                                  'transport_streaming': True,
                                  'api_timeout_seconds': args.api_timeout,
                                  'memory_window_size': args.memory_window_size,
                                  'planning': True, 'reflection': True, 'max_reflection_rounds': 2,
                                  'parallel_memory_build': False, 'parallel_retrieval': False},
            'uncertainty_scope': 'Trials repeat queries on a fixed memory bank; memory-build variance is excluded.'}
    directory = ROOT / 'reproduction' / 'results' / args.run_name
    directory.mkdir(parents=True, exist_ok=True)
    if args.mode == 'dry-run':
        save_json(directory / 'plan.json', plan)
        print(json.dumps({'status': 'dry_run_ok', 'questions': len(questions), 'trials': args.trials,
                          'planned_query_runs': len(questions) * args.trials * 2,
                          'api_calls': 0, 'dataset_sha256': provenance['sha256']}, indent=2))
        return
    if args.mode == 'import-check':
        from main import SimpleMemSystem
        print('Upstream text pipeline imports successfully. No model or API request was made.')
        return
    if not args.execute:
        parser.error('API execution requires --execute; default operation is a dry run')
    # No tool reads or prints the user's secret file. Only this application loads
    # the key for authentication; both output streams redact its exact value.
    import config
    if not config.OPENAI_API_KEY:
        raise ValueError('Local API key is not configured')
    sys.stdout = RedactingStream(sys.stdout, config.OPENAI_API_KEY)
    sys.stderr = RedactingStream(sys.stderr, config.OPENAI_API_KEY)
    if args.console_log:
        logfile = (directory / 'console.log').open('a', encoding='utf-8', buffering=1)
        # Tee is inside both redactors: raw values never reach the log file.
        sys.stdout.stream = TeeStream(sys.stdout.stream, logfile)
        sys.stderr.stream = TeeStream(sys.stderr.stream, logfile)
    if config.OPENAI_BASE_URL.rstrip('/') != 'https://api.deepseek.com':
        raise ValueError('Expected the authorized official DeepSeek endpoint')
    if config.LLM_MODEL != 'deepseek-flash':
        raise ValueError('Only deepseek-flash is covered by this budget guard')
    os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    embedding_manifest = ROOT / 'reproduction' / 'data' / 'embedding_provenance.json'
    local_embedding = ROOT / '.cache' / 'models' / 'Qwen3-Embedding-0.6B'
    if embedding_manifest.exists() and (local_embedding / 'model.safetensors').exists():
        os.environ['EMBEDDING_MODEL'] = str(local_embedding)
        plan['embedding_provenance'] = json.loads(embedding_manifest.read_text(encoding='utf-8'))
    else:
        raise ValueError('Run reproduction/download_embedding.py first to prepare the pinned original embedding model')
    meter = Meter(ROOT / 'reproduction' / 'results' / 'api_usage.jsonl', args.budget_cny, args.max_output,
                  timeout=args.api_timeout)
    plan.update({'model_requested': config.LLM_MODEL, 'provider': 'DeepSeek',
                 'base_url': config.OPENAI_BASE_URL, 'budget_guard_cny': args.budget_cny,
                 'price_source': 'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
                 'price_guard_cny_per_million': {'input': 2, 'output': 8},
                 'price_checked': '2026-11-08'})
    if (directory / 'plan.json').exists():
        old = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
        for field in ['questions', 'dataset', 'common_conditions', 'trials', 'retrieval_index']:
            if old[field] != plan[field]:
                raise ValueError(f'Existing run has different {field}; use another run name')
    save_json(directory / 'plan.json', plan)
    if args.mode == 'smoke':
        system = initialize(directory / 'bank', 'smoke', meter)
        from simplemem.core.models.memory_entry import Dialogue
        dialogues = [Dialogue(dialogue_id=1, speaker='Alice', content='Bob, let us meet at Starbucks on November 16, 2025 at 2 pm.', timestamp='2025-11-15T14:30:00'),
                     Dialogue(dialogue_id=2, speaker='Bob', content='Agreed. I will bring the market research report.', timestamp='2025-11-15T14:31:00')]
        built = ensure_memories(system, dialogues, directory / 'smoke_memory.json',
                                {'synthetic': True, 'model': config.LLM_MODEL}, meter)
        question = {'sample_index': -1, 'qa_index': 0, 'category': 4,
                    'question': 'Where will Alice and Bob meet?', 'answer': 'Starbucks'}
        row = score_question(system, question, 'smoke_only', 1, meter)
        save_json(directory / 'smoke_result.json', {'status': 'smoke_only_not_a_benchmark_result',
                  'memory': built, 'result': row, 'cumulative_peak_cny_upper_estimate': meter.reserved_cny})
        print(json.dumps({'status': 'smoke_completed', 'f1': row['f1'],
                          'peak_cny_upper_estimate': meter.reserved_cny}, indent=2))
        return
    rows_path = directory / 'answers.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text(encoding='utf-8').splitlines() if line] if rows_path.exists() else []
    completed = {(row['sample_index'], row['qa_index'], row['trial'], row['arm']) for row in rows}
    memory_summaries = []
    for sample_index in args.samples:
        sample_questions = [question for question in questions if question['sample_index'] == sample_index]
        system = initialize(directory / 'bank', f'sample_{sample_index}', meter)
        if system.memory_builder.window_size != args.memory_window_size:
            raise ValueError('Actual memory window differs from the requested protocol; inspect local configuration')
        dialogues = convert_dialogues(data[sample_index])
        identity = {'dataset_sha256': provenance['sha256'], 'sample_index': sample_index,
                    'model': config.LLM_MODEL, 'conditions': plan['common_conditions'],
                    'source_sha256': plan['environment']['source_sha256']}
        if args.cache_memory_windows:
            if __package__:
                from .memory_window_cache import install_memory_window_cache
            else:
                from memory_window_cache import install_memory_window_cache
            install_memory_window_cache(system, meter, ROOT / '.cache' / 'memory_window_replay', identity)
        built = ensure_memories(system, dialogues, directory / f'memory_{sample_index}.json', identity, meter)
        memory_summaries.append(built)
        refresh_full_text_index(system, directory / f'index_{sample_index}.json')
        for trial in range(1, args.trials + 1):
            schedule = [(question, arm, k) for question in sample_questions for arm, k in plan['arms'].items()]
            random.Random(args.seed + trial + sample_index * 1000).shuffle(schedule)
            for question, arm, k in schedule:
                key = (sample_index, question['qa_index'], trial, arm)
                if key in completed:
                    continue
                system.hybrid_retriever.semantic_top_k = k
                print(f'\n[Experiment] trial={trial} sample={sample_index} qa={question["qa_index"]} arm={arm} semantic_top_k={k}', flush=True)
                row = score_question(system, question, arm, trial, meter)
                with rows_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                rows.append(row)
                completed.add(key)
                summary = {'complete': len(rows) == len(questions) * args.trials * 2,
                           'arms': summarize(rows, args.trials, len(questions)),
                           'memory': memory_summaries, 'cumulative_peak_cny_upper_estimate': meter.reserved_cny}
                save_json(directory / 'summary.json', summary)
                print(f"Completed {len(rows)}/{len(questions) * args.trials * 2} query runs; guard total CNY {meter.reserved_cny:.4f}", flush=True)


if __name__ == '__main__':
    try:
        main()
    except BudgetStop as error:
        print(f'BUDGET_OR_VALIDITY_STOP: {error}', file=sys.stderr)
        raise SystemExit(2)
