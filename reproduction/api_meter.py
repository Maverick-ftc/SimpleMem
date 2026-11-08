"""Local, sequential API metering shared by memory construction and both arms.

No prompts, answers, keys, or Authorization headers are written to this ledger.
The guard reserves a conservative token upper bound before every request and uses
peak CNY prices (2 input / 8 output per million) regardless of time/cache savings.
Unknown failed requests remain charged at the reserved amount. There are no
automatic SDK retries, so potentially billable attempts cannot disappear.
"""
import json
import time
from pathlib import Path
from types import SimpleNamespace


class BudgetStop(BaseException):
    """Must escape upstream except Exception handlers and fallback logic."""


class Meter:
    def __init__(self, ledger, limit_cny=1.0, max_output=4096, transport_streaming=True, timeout=90):
        self.path = Path(ledger)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limit_cny = limit_cny
        self.max_output = max_output
        self.transport_streaming = transport_streaming
        self.timeout = timeout
        self.failure_streak = 0
        self.label = "initialization"
        self.records = []
        if self.path.exists():
            self.records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    @property
    def reserved_cny(self):
        return sum(record.get("guard_cny", 0) for record in self.records)

    def write(self, record):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.append(record)

    def wrap(self, llm_client):
        from openai import OpenAI
        real = OpenAI(api_key=llm_client.api_key, base_url=llm_client.base_url,
                      max_retries=0, timeout=self.timeout)

        def create(**kwargs):
            if kwargs.get("stream"):
                raise ValueError("The adapter manages transport streaming; the agent must request a complete response")
            if kwargs["model"] != "deepseek-flash":
                raise ValueError("Price guard is configured only for deepseek-flash")
            kwargs["max_tokens"] = self.max_output
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            # UTF-8 bytes upper-bound byte-level text tokenization; extra margin
            # covers message wrappers. This is a guard, not a usage estimate.
            input_bound = len(json.dumps(kwargs["messages"], ensure_ascii=False).encode("utf-8")) + 4096
            reservation = (input_bound * 2 + self.max_output * 8) / 1_000_000
            if self.reserved_cny + reservation > self.limit_cny:
                raise BudgetStop(f"Local budget guard stopped before request: cap CNY {self.limit_cny:.2f}")
            request_id = len(self.records)
            self.write({"id": request_id, "event": "reservation", "label": self.label,
                        "time": time.time(), "guard_cny": reservation})
            started = time.perf_counter()
            received = 0
            try:
                if self.transport_streaming:
                    kwargs['stream'] = True
                    kwargs['stream_options'] = {'include_usage': True}
                    content, usage, finish_reason, model = [], None, None, kwargs['model']
                    received, next_notice = 0, 5000
                    with real.chat.completions.create(**kwargs) as stream:
                        for chunk in stream:
                            model = chunk.model or model
                            if chunk.usage is not None:
                                usage = chunk.usage
                            for choice in chunk.choices:
                                text = choice.delta.content or ''
                                content.append(text)
                                received += len(text)
                                if received >= next_notice:
                                    print(f'[API stream] {self.label}: received {received} characters', flush=True)
                                    next_notice += 5000
                                if choice.finish_reason is not None:
                                    finish_reason = choice.finish_reason
                            # A terminal reason plus final usage is complete; do
                            # not wait for an extra transport sentinel to arrive.
                            if finish_reason is not None and usage is not None:
                                break
                    response = SimpleNamespace(model=model, usage=usage, choices=[SimpleNamespace(
                        finish_reason=finish_reason, message=SimpleNamespace(content=''.join(content)))])
                else:
                    response = real.chat.completions.create(**kwargs)
            except BaseException as error:
                causes, cause = [], error.__cause__
                while cause is not None and len(causes) < 5:
                    causes.append(type(cause).__name__)
                    cause = cause.__cause__
                self.write({"id": request_id, "event": "error", "error_type": type(error).__name__,
                            "cause_types": causes, "received_characters": received,
                            "seconds": time.perf_counter() - started, "guard_cny": 0})
                print(f'[API error] {type(error).__name__}; causes={causes}; received_characters={received}', flush=True)
                self.failure_streak += 1
                if self.failure_streak >= 3:
                    raise BudgetStop('Three consecutive API failures; stopping before upstream can silently fall back')
                raise
            self.failure_streak = 0
            usage = response.usage
            if usage is None:
                raise BudgetStop("Missing provider usage; stopping with reservation retained")
            prompt, completion = usage.prompt_tokens, usage.completion_tokens
            actual_peak = (prompt * 2 + completion * 8) / 1_000_000
            self.write({"id": request_id, "event": "response", "label": self.label,
                        "model": response.model, "usage": usage.model_dump(),
                        "temperature": kwargs.get("temperature"), "max_tokens": self.max_output,
                        "thinking": "disabled",
                        "transport_streaming": self.transport_streaming,
                        "timeout_seconds": self.timeout,
                        "finish_reason": response.choices[0].finish_reason,
                        "seconds": time.perf_counter() - started,
                        "guard_cny": actual_peak - reservation})
            if response.choices[0].finish_reason == "length":
                raise BudgetStop("Truncated response: stopping rather than reporting an invalid result")
            if response.choices[0].finish_reason != 'stop':
                raise BudgetStop('Response did not finish with a normal stop')
            if not response.choices[0].message.content:
                raise BudgetStop("Empty model response: stopping")
            return response

        llm_client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
