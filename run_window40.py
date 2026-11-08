"""Controlled rerun with the released 40-turn memory window.

Run this file in PyCharm. Existing 10-turn results remain separate.
Successful parsed memory windows are replayable after a connection interruption.
"""
import sys
from reproduction.run_experiment import main
from reproduction.api_meter import BudgetStop

RUN_NAME = 'locomo_window40_20261108'
MEMORY_WINDOW_SIZE = 40
CUMULATIVE_BUDGET_CNY = 15

if __name__ == '__main__':
    sys.argv = [__file__, '--mode', 'experiment', '--execute',
                '--samples', '0', '--per-category', '2', '--trials', '3',
                '--run-name', RUN_NAME, '--memory-window-size', str(MEMORY_WINDOW_SIZE),
                '--max-output', '16384', '--api-timeout', '180',
                '--budget-cny', str(CUMULATIVE_BUDGET_CNY),
                '--cache-memory-windows', '--console-log']
    try:
        main()
    except BudgetStop as error:
        print(f'BUDGET_OR_VALIDITY_STOP: {error}', flush=True)
        raise SystemExit(2)
