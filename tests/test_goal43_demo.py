from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from goal43_demo import write_demo


def payload():
    return {
        "schema_version": 1,
        "title": "Метро: GOAL 4.3",
        "scenarios": [{
            "scenario_id": "station_stop_8",
            "title": "Остановка на 8 минут",
            "evaluation_mode": "historical_demand_synthetic_incident",
            "incident": {"station_id": "L1_08", "duration_minutes": 8},
            "decisions": [{"reason": "Остановка состава", "action": "REGULATE_ONLY"}],
            "comparisons": [{
                "strategy": "REGULATE_ONLY", "status": "checked", "elapsed_seconds": 3.2,
                "metrics": {"train_km": 240.5, "saving_km": 12.3,
                            "max_car_load": 150, "exceptional_time_share": 0.0},
                "per_train": [{"train_id": "B001", "train_type": "Baltiets",
                               "baseline_km": 252.8, "candidate_km": 240.5, "saving_km": 12.3}],
                "series": {"load": [{"second": 0, "load": 120}, {"second": 900, "load": 150}],
                           "queue": [{"second": 0, "queued": 12}],
                           "headway": [{"second": 0, "seconds": 150}],
                           "entries": [{"second": 0, "actual": 250, "forecast": 230}],
                           "fleet": [{"second": 0, "count": 53}]},
            }],
        }],
    }


def embedded_payload(path):
    document = path.read_text(encoding="utf-8")
    match = re.search(r'<script id="demo-data" type="application/json">(.*?)</script>', document, re.S)
    assert match
    return json.loads(match.group(1)), document


def test_demo_is_standalone_local_html_with_all_views(tmp_path):
    result = write_demo(tmp_path / "demo", payload())
    assert result == tmp_path / "demo" / "demo.html"
    data, html = embedded_payload(result)
    assert data["scenarios"][0]["comparisons"][0]["metrics"]["saving_km"] == 12.3
    for identifier in ("scenario-select", "strategy-select", "comparison-table", "per-train-table",
                       "load-chart", "queue-chart", "headway-chart", "entries-chart", "fleet-chart"):
        assert f'id="{identifier}"' in html
    offline_html = html.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in offline_html and "https://" not in offline_html
    assert not re.search(r'<script[^>]+src=|<link[^>]+href=', html)
    assert "innerHTML" not in html
    assert "Воспроизведение сохранённого расчёта" in html
    assert "Время сохранённого расчёта" in html
    assert "Управление выпуском отключено" in html
    assert data["release_authorized"] is False


def test_user_text_cannot_break_out_of_embedded_json_script(tmp_path):
    data = payload()
    attack = '</script><script>alert("x")</script><img src=x onerror=alert(1)>&\u2028\u2029'
    data["title"] = attack
    data["scenarios"][0]["title"] = attack
    data["scenarios"][0]["comparisons"][0]["per_train"][0]["train_id"] = attack
    original = deepcopy(data)
    saved, html = embedded_payload(write_demo(tmp_path, data))
    assert saved["title"] == attack
    assert data == original
    assert attack not in html
    assert html.count("</script>") == 2
    assert "\\u003c/script\\u003e" in html
    assert "\\u2028" in html and "\\u2029" in html
    assert "textContent" in html


def test_missing_optional_metrics_and_series_remain_unknown(tmp_path):
    data = {"schema_version": 1, "scenarios": [{"scenario_id": "minimal", "comparisons": [{}]}]}
    saved, html = embedded_payload(write_demo(tmp_path, data))
    assert saved["scenarios"][0]["comparisons"] == [{}]
    assert "Неизвестно" in html
    assert "Нет данных" in html
    assert "Нет сохранённых сценариев" in html


def test_empty_payload_produces_openable_demo(tmp_path):
    saved, _ = embedded_payload(write_demo(tmp_path, {}))
    assert saved["scenarios"] == []
    assert saved["schema_version"] == 1


def test_existing_repository_schema_string_is_accepted(tmp_path):
    data = payload()
    data["schema_version"] = "1.0"
    saved, _ = embedded_payload(write_demo(tmp_path, data))
    assert saved["schema_version"] == 1


@pytest.mark.parametrize("data", [
    {"schema_version": 2},
    {"scenarios": "invalid"},
    {"scenarios": [{"scenario_id": "same"}, {"scenario_id": "same"}]},
    {"release_authorized": True},
])
def test_invalid_or_live_dispatch_payload_is_rejected(tmp_path, data):
    with pytest.raises(ValueError):
        write_demo(tmp_path, data)
    assert not (tmp_path / "demo.html").exists()


def test_nonfinite_numbers_are_not_serialized_as_fake_json(tmp_path):
    data = payload()
    data["scenarios"][0]["comparisons"][0]["metrics"]["saving_km"] = float("nan")
    with pytest.raises(ValueError):
        write_demo(tmp_path, data)


def test_n_values_and_live_latency_are_kept_separate(tmp_path):
    data = payload()
    data["scenarios"] = [dict(data["scenarios"][0], scenario_id=f"stop_{n}",
                              title=f"N={n}", incident={"duration_minutes": n}) for n in (3, 8, 15, 30)]
    saved, html = embedded_payload(write_demo(tmp_path, data))
    assert [row["incident"]["duration_minutes"] for row in saved["scenarios"]] == [3, 8, 15, 30]
    assert "480" in html
    assert "Время открытия страницы не является временем поиска" in html


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is optional for JavaScript checks")
def test_nested_runner_selection_and_viewed_alternative_remain_distinct(tmp_path):
    data = payload()
    row = dict(selection=dict(status="conditional_resource_policy", selected_candidate_id="KEEP"),
               candidate_table=[dict(candidate_id="KEEP", action=dict(kind="KEEP")),
                                dict(candidate_id="WITHDRAW_ONLY:B001", action=dict(kind="WITHDRAW_ONLY",
                                     actions=[dict(train_id="B001", actual_effect_second=1000)]))])
    _, html = embedded_payload(write_demo(tmp_path, data))
    js = re.search(r'// decision-descriptor:start\n(.*?)// decision-descriptor:end', html, re.S)
    assert js
    script = js.group(1) + "console.log(JSON.stringify(describeDecision(JSON.parse(process.argv[1]),JSON.parse(process.argv[2]))));"
    result = subprocess.run(["node", "-e", script, json.dumps(row),
                             json.dumps(dict(candidate_id="WITHDRAW_ONLY:B001"))],
                            capture_output=True, text=True, check=True)
    descriptor = json.loads(result.stdout)
    assert descriptor["recommended_candidate_id"] == "KEEP"
    assert descriptor["viewed_candidate_id"] == "WITHDRAW_ONLY:B001"
    assert descriptor["action"]["kind"] == "WITHDRAW_ONLY"
    assert descriptor["action"]["actions"][0]["actual_effect_second"] == 1000


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is optional for JavaScript checks")
def test_injected_historical_incident_and_fixture_are_labeled_synthetic(tmp_path):
    _, html = embedded_payload(write_demo(tmp_path, payload()))
    function = re.search(r"function modeLabel\(mode\)\{[^\n]+", html).group()
    script = "const txt=value=>String(value);" + function + "console.log(JSON.stringify([modeLabel(process.argv[1]),modeLabel('fixture')]));"
    result = subprocess.run(["node", "-e", script, "historical_entries_injected_incident_assumed_timetable"],
                            capture_output=True, text=True, encoding="utf-8", check=True)
    historical, fixture = json.loads(result.stdout)
    assert "историчес" in historical.lower() and "искусствен" in historical.lower()
    assert "искусствен" in fixture.lower()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is optional for JavaScript checks")
def test_rolling_view_uses_each_saved_recommendation_instead_of_missing_candidate(tmp_path):
    _, html = embedded_payload(write_demo(tmp_path, payload()))
    function = re.search(r'// decision-descriptor:start\n(.*?)// decision-descriptor:end', html, re.S).group(1)
    row = dict(selection=dict(selected_candidate_id="REGULATE_ONLY"), candidate_table=[
        dict(candidate_id="REGULATE_ONLY", action=dict(kind="REGULATE_ONLY"))])
    script = function + "console.log(JSON.stringify(describeDecision(JSON.parse(process.argv[1]),{candidate_id:'ROLLING_SELECTED'})));"
    result = subprocess.run(["node", "-e", script, json.dumps(row)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["action"]["kind"] == "REGULATE_ONLY"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is optional for JavaScript checks")
def test_timetable_change_and_depot_return_are_separate_user_visible_effects(tmp_path):
    _, html = embedded_payload(write_demo(tmp_path, payload()))
    function = re.search(r'// effect-lines:start\n(.*?)// effect-lines:end', html, re.S).group(1)
    action = dict(kind="WITHDRAW_ONLY", timetable_change_second=100, timetable_change_by_station={"A": 100},
        guaranteed_improvement=False, actions=[dict(train_id="B001", depot_return_second=500, actual_effect_second=500)])
    script = "const finite=v=>typeof v==='number',fmt=(v,u)=>v+' '+u,txt=String;" + function
    script += "console.log(JSON.stringify(effectLines(JSON.parse(process.argv[1]),{decision_second:0})));"
    result = subprocess.run(["node", "-e", script, json.dumps(action)], capture_output=True, text=True, encoding="utf-8", check=True)
    lines = json.loads(result.stdout)
    assert any("график" in line and "100" in line for line in lines)
    assert any("депо" in line and "500" in line for line in lines)
    assert 'id="od-table"' in html


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is optional for JavaScript checks")
def test_headway_aggregation_label_is_visible_and_accessible_in_the_svg(tmp_path):
    data = payload()
    series = data["scenarios"][0]["comparisons"][0]["series"]
    series["headway_label"] = "Максимальный интервал по направлениям за минуту"
    _, html = embedded_payload(write_demo(tmp_path, data))
    functions = re.search(r'(function svgNode.*?)function renderProvenance', html, re.S).group(1)
    call = re.search(r"chart\('headway-chart'.*?;", html).group()
    stub = """
class Element {constructor(tag){this.tag=tag;this.attrs={};this.children=[];this.parentElement={querySelector:()=>({textContent:'Интервалы движения'})};}
setAttribute(k,v){this.attrs[k]=v;}append(x){this.children.push(x);}replaceChildren(){this.children=[];}}
const target=new Element('div'),el=()=>target,array=v=>Array.isArray(v)?v:[],finite=Number.isFinite;
const document={createElementNS:(_,tag)=>new Element(tag)},number=String,fmt=String;
const node=(tag,text)=>{const value=new Element(tag);value.textContent=text;return value;};
const series=JSON.parse(process.argv[1]);
"""
    script = stub + functions + call + "console.log(JSON.stringify(target));"
    result = subprocess.run(["node", "-e", script, json.dumps(series)], capture_output=True,
                            text=True, encoding="utf-8", check=True)
    rendered = json.loads(result.stdout)
    label = series["headway_label"]
    svg = rendered["children"][0]
    assert label in svg["attrs"]["aria-label"]
    assert any(child.get("tag") == "title" and label in child["textContent"] for child in svg["children"])
    assert label in rendered["children"][1]["textContent"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is optional for JavaScript checks")
def test_incident_summary_keeps_actual_duration_separate_from_the_initial_estimate(tmp_path):
    _, html = embedded_payload(write_demo(tmp_path, payload()))
    function = re.search(r'// incident-duration:start\n(.*?)// incident-duration:end', html, re.S)
    assert function
    script = "const finite=Number.isFinite,fmt=(v,u)=>v+' '+u;" + function.group(1)
    script += "console.log(incidentDurationLabel({actual_duration_minutes:15,initial_estimated_duration_seconds:300}));"
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, encoding="utf-8", check=True)
    assert "факт" in result.stdout and "15" in result.stdout and "оценка" in result.stdout and "5" in result.stdout
    assert "Сокращение на состав (фиксированные 68)" in html


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is optional for JavaScript checks")
def test_audit_text_distinguishes_full_day_window_failures_and_queue_from_waiting_cohort(tmp_path):
    _, html = embedded_payload(write_demo(tmp_path, payload()))
    function = re.search(r'// service-audit:start\n(.*?)// service-audit:end', html, re.S)
    assert function
    script = "const array=x=>Array.isArray(x)?x:[],finite=Number.isFinite,fmt=(v,u)=>v+' '+u,txt=String;" + function.group(1)
    script += "console.log(JSON.stringify(serviceAuditLines({audit_full_day:true,od_results:[{od:'S1',queued:49,maximum_window_share:0.02,failed_15m_windows:[{}],cohort_scope:{start_second:0,end_second:1800}}]})));"
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, encoding="utf-8", check=True)
    assert "полная смена" in result.stdout and "49" in result.stdout and "15" in result.stdout and "1800" in result.stdout
