import copy
import json
import time
from types import SimpleNamespace

import pytest

from metro51_explanations import (
    ExplanationError, YandexExplanationProvider, build_fact_packet,
    explain_decision, load_env, main, validate_explanation, validate_selection,
)


def packet():
    return build_fact_packet(
        run_id='test-run', decision_id='08:00',
        action={'type': 'KEEP', 'train_ids': ['T1']},
        facts=[
            {'id': 'forecast.A', 'label': 'Прогноз входов A', 'value': 1200,
             'unit': 'человек', 'source': 'forecasts.json#A', 'kind': 'forecast'},
            {'id': 'limit.headway', 'label': 'Минимальный интервал', 'value': 113,
             'unit': 'секунд', 'source': 'constraints.json', 'kind': 'constraint'},
            {'id': 'risk.A', 'label': 'Непосадки на A', 'value': .04,
             'unit': 'доля', 'source': 'audit.json#A', 'kind': 'risk'},
            {'id': 'reason.keep', 'label': 'Основание решения',
             'value': 'Снятие отклонено из-за нарушения локального ограничения.',
             'unit': '', 'source': 'candidate_checks.json', 'kind': 'rationale'},
        ],
        model_context={'model_id': 'stack-v1', 'attribution_available': False},
    )


def selection(p):
    return {
        'run_id': p['run_id'], 'decision_id': p['decision_id'],
        'context_hash': p['context_hash'],
        'sections': {'action': ['decision.action'], 'prediction': ['forecast.A'],
                     'rationale': ['reason.keep', 'limit.headway'], 'effects': [],
                     'risks': ['risk.A'], 'revisit': []},
    }


class FakeProvider:
    model = 'test-model'

    def __init__(self, response=None, delay=0, error=None):
        self.response, self.delay, self.error, self.calls = response, delay, error, 0

    def generate_structured(self, **kwargs):
        self.calls += 1
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return copy.deepcopy(self.response)


def test_packet_determinism_and_input_isolation():
    action = {'type': 'KEEP'}
    p = build_fact_packet(run_id='r', decision_id='d', action=action, facts=[])
    action['type'] = 'WITHDRAW'
    assert p['action']['type'] == 'KEEP'
    assert p['context_hash'] == build_fact_packet(run_id='r', decision_id='d', action={'type': 'KEEP'}, facts=[])['context_hash']
    assert p['context_hash'] != build_fact_packet(run_id='r', decision_id='d2', action={'type': 'KEEP'}, facts=[])['context_hash']


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), object()])
def test_packet_rejects_non_json_and_non_finite_values(bad):
    with pytest.raises(ExplanationError):
        build_fact_packet(run_id='r', decision_id='d', action={'type': 'KEEP'},
                          facts=[dict(id='x', label='x', value=bad, unit='', source='s')])


def test_packet_rejects_duplicates_and_credentials():
    f = dict(id='x', label='x', value=1, unit='', source='s')
    with pytest.raises(ExplanationError, match='duplicate'):
        build_fact_packet(run_id='r', decision_id='d', action={}, facts=[f, f])
    with pytest.raises(ExplanationError, match='credential'):
        build_fact_packet(run_id='r', decision_id='d', action={}, facts=[], model_context={'api_key': 'secret'})


def test_packet_cannot_be_modified_after_hash():
    p = packet()
    p['action']['type'] = 'WITHDRAW'
    with pytest.raises(ExplanationError, match='hash'):
        explain_decision(p, use_api=False)


def test_valid_llm_selection_preserves_plan_and_numbers():
    p = packet()
    original = copy.deepcopy(p)
    provider = FakeProvider(selection(p))
    result = explain_decision(p, provider=provider)
    assert p == original
    assert result['action'] == p['action']
    assert result['metadata']['source'] == 'yandexgpt'
    assert result['metadata']['validated'] is True
    assert '1200' in result['sections']['prediction'][0]['text']
    assert '113' in result['sections']['rationale'][1]['text']
    assert 'без измеренных вкладов' in result['limitations'][0]


@pytest.mark.parametrize('field,value', [('run_id', 'other'), ('decision_id', '09:00'), ('context_hash', 'old')])
def test_wrong_context_rejected(field, value):
    p = packet()
    answer = selection(p)
    answer[field] = value
    with pytest.raises(ExplanationError):
        validate_selection(p, answer)


def test_hallucinated_facts_numbers_prose_and_action_rejected():
    p = packet()
    for update in [{'text': 'Экономия 999999'}, {'action': {'type': 'WITHDRAW'}}]:
        answer = selection(p)
        answer.update(update)
        with pytest.raises(ExplanationError):
            validate_selection(p, answer)
    answer = selection(p)
    answer['sections']['rationale'] = ['weather.caused_growth']
    result = explain_decision(p, provider=FakeProvider(answer))
    assert result['metadata']['source'] == 'template'
    assert result['metadata']['error_code'] == 'invalid_response'
    assert 'weather.caused_growth' not in json.dumps(result)


def test_attribution_cannot_be_inferred_from_forecast():
    p = packet()
    answer = selection(p)
    answer['sections']['rationale'] = ['forecast.A']
    with pytest.raises(ExplanationError, match='kind'):
        validate_selection(p, answer)


def test_timeout_returns_fast_and_late_result_is_not_cached():
    p, cache = packet(), {}
    provider = FakeProvider(selection(p), delay=.15)
    start = time.monotonic()
    result = explain_decision(p, provider=provider, timeout_s=.01, cache=cache)
    assert time.monotonic() - start < .1
    assert result['metadata']['error_code'] == 'timeout'
    time.sleep(.18)
    assert cache == {}


def test_cache_is_context_and_model_specific():
    p, cache = packet(), {}
    provider = FakeProvider(selection(p))
    a = explain_decision(p, provider=provider, cache=cache)
    b = explain_decision(p, provider=provider, cache=cache)
    assert provider.calls == 1
    assert b['metadata']['cache_hit'] is True
    b['action']['type'] = 'WITHDRAW'
    assert a['action']['type'] == 'KEEP'
    provider.model = 'another-model'
    explain_decision(p, provider=provider, cache=cache)
    assert provider.calls == 2


def test_disabled_api_and_errors_use_verified_template_without_secrets():
    p = packet()
    provider = FakeProvider(error=RuntimeError('YANDEX_API_KEY=secret-value'))
    disabled = explain_decision(p, provider=provider, use_api=False)
    assert provider.calls == 0 and disabled['metadata']['source'] == 'template'
    failed = explain_decision(p, provider=provider)
    assert failed['metadata']['error_code'] == 'provider_error'
    assert 'secret-value' not in json.dumps(failed)


def test_load_root_env_preserves_shell_values_and_ignores_other_secrets(tmp_path):
    env = tmp_path / '.env'
    env.write_text('YANDEX_API_KEY="private"\nYANDEX_FOLDER_ID=folder\nYANDEX_MODEL=yandexgpt-5-lite\nOTHER_TOKEN=hidden\n', encoding='utf-8')
    target = {'YANDEX_API_KEY': 'shell'}
    assert load_env(env, environ=target) == ['YANDEX_FOLDER_ID', 'YANDEX_MODEL']
    assert target['YANDEX_API_KEY'] == 'shell'
    assert 'OTHER_TOKEN' not in target


def test_schema_only_fallback_and_no_model_substitution():
    class StatusError(Exception):
        status_code = 400
        body = {'error': 'json_schema response_format not supported'}

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise StatusError()
            message = SimpleNamespace(content=json.dumps(selection(packet())), refusal=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason='stop')], usage=None)

    client = Client()
    provider = YandexExplanationProvider(environ={'YANDEX_API_KEY': 'key', 'YANDEX_FOLDER_ID': 'folder',
                                                 'YANDEX_MODEL': 'yandexgpt-5-lite'}, client=client)
    response = provider.generate_structured(packet=packet(), timeout_s=1)
    assert response['run_id'] == 'test-run'
    assert len(client.calls) == 2
    assert all(call['model'] == 'gpt://folder/yandexgpt-5-lite' for call in client.calls)
    assert client.calls[1]['response_format']['type'] == 'json_object'


@pytest.mark.parametrize('status', [400, 401, 403, 429, 500])
def test_api_errors_are_sanitized_and_do_not_retry(status):
    class StatusError(Exception):
        status_code = status
        body = {'error': 'secret model private details'}

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise StatusError('key value')

    client = Client()
    provider = YandexExplanationProvider(environ={'YANDEX_API_KEY': 'key', 'YANDEX_FOLDER_ID': 'folder',
                                                 'YANDEX_MODEL': 'yandexgpt-5-lite'}, client=client)
    with pytest.raises(ExplanationError) as error:
        provider.generate_structured(packet=packet(), timeout_s=1)
    assert error.value.code == f'http_{status}'
    assert client.calls == 1
    assert 'secret' not in str(error.value) and 'key value' not in str(error.value)


def test_unsupported_model_records_model_code_not_raw_body():
    class StatusError(Exception):
        status_code = 404
        body = {'error': 'model yandexgpt-5-lite not found; key private'}

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(StatusError()))))
    provider = YandexExplanationProvider(environ={'YANDEX_API_KEY': 'key', 'YANDEX_FOLDER_ID': 'folder',
                                                 'YANDEX_MODEL': 'yandexgpt-5-lite'}, client=client)
    result = explain_decision(packet(), provider=provider)
    assert result['metadata']['error_code'] == 'model_unavailable'
    assert result['metadata']['http_status'] == 404
    assert result['metadata']['model'] == 'yandexgpt-5-lite'


def test_cli_template_report_is_reproducible_and_creates_parent(tmp_path, capsys):
    source, output = tmp_path / 'packet.json', tmp_path / 'reports' / 'result.json'
    source.write_text(json.dumps(packet(), ensure_ascii=False), encoding='utf-8')
    assert main(['--packet', str(source), '--output', str(output), '--no-api']) == 0
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['metadata']['source'] == 'template'
    assert 'YANDEX_API_KEY' not in capsys.readouterr().out


def test_real_sdk_transport_header_schema_and_literal_model():
    import httpx
    from openai import OpenAI

    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion',
            'created': 0, 'model': 'gpt://folder/yandexgpt-5-lite',
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                'role': 'assistant', 'content': json.dumps(selection(packet()))}}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 8, 'total_tokens': 20}})

    client = OpenAI(api_key='test-key', project='folder',
                    base_url='https://ai.api.cloud.yandex.net/v1',
                    default_headers={'Authorization': 'Api-Key test-key'}, max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    provider = YandexExplanationProvider(environ={'YANDEX_API_KEY': 'test-key', 'YANDEX_FOLDER_ID': 'folder',
                                                 'YANDEX_MODEL': 'yandexgpt-5-lite'}, client=client)
    try:
        result = explain_decision(packet(), provider=provider)
    finally:
        provider.close()
    assert result['metadata']['source'] == 'yandexgpt'
    assert result['metadata']['usage']['total_tokens'] == 20
    assert str(requests[0].url) == 'https://ai.api.cloud.yandex.net/v1/chat/completions'
    assert requests[0].headers['Authorization'] == 'Api-Key test-key'
    assert requests[0].headers['OpenAI-Project'] == 'folder'
    body = json.loads(requests[0].content)
    assert body['model'] == 'gpt://folder/yandexgpt-5-lite'
    assert body['response_format']['type'] == 'json_schema'
    assert 'test-key' not in body['messages'][0]['content'] + body['messages'][1]['content']


def test_saved_explanation_validation_detects_modified_prose_plan_and_evidence():
    p = packet()
    result = explain_decision(p, use_api=False)
    assert validate_explanation(p, result) is True
    changed = copy.deepcopy(result)
    changed['sections']['prediction'][0]['text'] += ' Экономия 999999 рублей.'
    with pytest.raises(ExplanationError):
        validate_explanation(p, changed)
    changed = copy.deepcopy(result)
    changed['action']['type'] = 'WITHDRAW'
    with pytest.raises(ExplanationError):
        validate_explanation(p, changed)
