"""Check the saved offline demo in headless Edge, without a browser framework."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import struct
import subprocess


EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
ERRORS = r'''<script id="goal43-ui-errors">
window.__goal43UIErrors=[];
addEventListener('error',event=>window.__goal43UIErrors.push(String(event.message)+' @ '+String(event.filename)+':'+event.lineno));
addEventListener('unhandledrejection',event=>window.__goal43UIErrors.push('Unhandled rejection: '+String(event.reason)));
</script>'''
SCRIPT = r'''<script id="goal43-ui-run">
(()=>{
const receipt={status:'running',scenario_count:0,view_count:0,views:[],errors:[],demo_data_removed:false};
const check=(ok,message)=>{if(!ok)throw Error(message);};
const select=(element,value)=>{element.value=String(value);element.dispatchEvent(new Event('change',{bubbles:true}));};
const ids=['entries-chart','load-chart','queue-chart','headway-chart','fleet-chart'];
try{
const payload=JSON.parse(document.getElementById('demo-data').textContent),scenarios=payload.scenarios||[];
const scenarioSelect=document.getElementById('scenario-select'),strategySelect=document.getElementById('strategy-select');
receipt.scenario_count=scenarios.length;
check(scenarioSelect.options.length===scenarios.length,'Scenario option count');
scenarios.forEach((scenario,index)=>{
select(scenarioSelect,index);const comparisons=scenario.comparisons||[];
check(strategySelect.options.length===comparisons.length,'Strategy option count for '+scenario.scenario_id);
comparisons.forEach((comparison,method)=>{
const view={scenario_id:scenario.scenario_id,candidate_id:comparison.candidate_id||comparison.strategy,passed:false};
try{
select(strategySelect,method);
const metrics=[...document.querySelectorAll('#metric-grid .metric')];
check(metrics.length>=10,'Missing metrics');
check(!metrics.some(row=>/NaN|Infinity|undefined/.test(row.textContent)),'Nonfinite metric text');
check(document.querySelectorAll('#comparison-table tbody tr').length===comparisons.length,'Comparison table rows');
check(document.querySelectorAll('#per-train-table tbody tr').length===Math.max(1,(comparison.per_train||[]).length),'Per-train table rows');
check(document.querySelectorAll('#comparison-table .selected-row').length===1,'Selected comparison row');
let svgCount=0;
ids.forEach(id=>{const target=document.getElementById(id);check(!!target,'Missing chart '+id);
target.querySelectorAll('svg').forEach(svg=>{svgCount++;
check(svg.getAttribute('aria-label'),'Missing SVG description');
const box=svg.getAttribute('viewBox').split(/\s+/).map(Number);
check(box.length===4&&box.every(Number.isFinite)&&box[2]>0&&box[3]>0,'Invalid SVG viewBox');
svg.querySelectorAll('*').forEach(element=>{for(const attr of element.attributes){
check(!/NaN|Infinity|undefined|null/.test(attr.value),'Invalid SVG '+attr.name);
if(['x','y','x1','x2','y1','y2','cx','cy','r'].includes(attr.name))check(Number.isFinite(Number(attr.value)),'Nonfinite SVG coordinate');
}});
if(id==='headway-chart'&&comparison.series&&comparison.series.headway_label)
check(svg.getAttribute('aria-label').includes(comparison.series.headway_label),'Headway aggregation label missing');
});});
view.metric_count=metrics.length;view.per_train_rows=(comparison.per_train||[]).length;view.svg_count=svgCount;view.passed=true;
}catch(error){view.error=String(error.stack||error);receipt.errors.push(view.scenario_id+'/'+view.candidate_id+': '+view.error);}
receipt.views.push(view);
});});
receipt.view_count=receipt.views.length;
const finalIndex=scenarios.findIndex(row=>row.scenario_id==='fixture_offpeak');
check(finalIndex>=0,'Screenshot fixture_offpeak missing');select(scenarioSelect,finalIndex);
const finalMethods=scenarios[finalIndex].comparisons||[],finalMethod=finalMethods.findIndex(row=>row.strategy==='ROLLING_SELECTED');
check(finalMethod>=0,'Screenshot ROLLING_SELECTED missing');select(strategySelect,finalMethod);
receipt.screenshot_scope={scenario_id:'fixture_offpeak',strategy:'ROLLING_SELECTED'};
receipt.viewport={width:innerWidth,height:innerHeight};
}catch(error){receipt.errors.push(String(error.stack||error));}
setTimeout(()=>{
receipt.errors.push(...window.__goal43UIErrors);
document.getElementById('demo-data').textContent='';receipt.demo_data_removed=true;
receipt.status=receipt.errors.length||receipt.views.some(row=>!row.passed)?'failed':'passed';
const output=document.createElement('pre');output.id='goal43-ui-receipt';output.hidden=true;output.textContent=JSON.stringify(receipt);document.body.append(output);
},0);
})();
</script>'''


def prepare_probe(output):
    output = Path(output).resolve()
    source = (output / "demo.html").read_text(encoding="utf-8")
    if source.count('<script id="demo-data"') != 1:
        raise ValueError("Expected one offline demo payload")
    source = source.replace('<script id="demo-data"', ERRORS + '\n<script id="demo-data"', 1)
    source = source.replace("</body>", SCRIPT + "\n</body>", 1)
    path = output / "verification/ui_probe.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


class _Receipt(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.capture, self.parts = False, []

    def handle_starttag(self, tag, attrs):
        if tag == "pre" and dict(attrs).get("id") == "goal43-ui-receipt":
            self.capture = True

    def handle_endtag(self, tag):
        if tag == "pre":
            self.capture = False

    def handle_data(self, data):
        if self.capture:
            self.parts.append(data)


def read_receipt(dom):
    parser = _Receipt()
    parser.feed(dom)
    if not parser.parts:
        raise ValueError("UI receipt was not produced")
    return json.loads("".join(parser.parts))


def _compact_dom(dom):
    return re.sub(r'''(<script\b[^>]*\bid=["']demo-data["'][^>]*>).*?(</script>)''',
                  r"\1\2", dom, flags=re.I | re.S)


def probe(output, *, edge_path=EDGE, timeout=120):
    output, edge_path = Path(output).resolve(), Path(edge_path).resolve()
    if not edge_path.is_file():
        raise FileNotFoundError(edge_path)
    source = output / "demo.html"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    page = prepare_probe(output)
    folder, screenshot = page.parent, page.parent / "preview.png"
    command = [str(edge_path), "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
        "--allow-file-access-from-files", "--window-size=1440,1600", "--force-device-scale-factor=1",
        "--virtual-time-budget=5000", f"--user-data-dir={folder / 'edge-ui-profile'}",
        f"--screenshot={screenshot}", "--dump-dom", page.as_uri()]
    try:
        process = subprocess.run(command, capture_output=True, timeout=timeout,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        dom = process.stdout.decode("utf-8", errors="replace")
        (folder / "ui_dom.html").write_text(_compact_dom(dom), encoding="utf-8")
        (folder / "ui_edge_stderr.log").write_bytes(process.stderr)
        receipt = read_receipt(dom)
        if process.returncode:
            receipt.update(status="failed", browser_exit_code=process.returncode)
        dimensions = list(struct.unpack(">II", screenshot.read_bytes()[16:24])) if screenshot.is_file() else None
        receipt["screenshot_dimensions"] = dimensions
        if dimensions != [1440, 1600]:
            receipt.setdefault("errors", []).append("Screenshot dimensions differ from 1440x1600")
            receipt["status"] = "failed"
    except (subprocess.TimeoutExpired, ValueError, OSError, struct.error) as error:
        receipt = dict(status="failed", errors=[str(error)])
    receipt.update(schema_version=1, browser_path=str(edge_path), source_html_sha256=source_hash,
        source_html_unchanged=hashlib.sha256(source.read_bytes()).hexdigest() == source_hash,
        generated_at=datetime.now(timezone.utc).isoformat(), screenshot=str(screenshot),
        scope="all saved scenario/method change events; finite DOM metrics, tables and SVG; offline file")
    if not receipt["source_html_unchanged"]:
        receipt["status"] = "failed"
    (folder / "ui_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    if receipt["status"] != "passed":
        raise ValueError("UI probe failed: " + "; ".join(receipt.get("errors", [])))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = probe(args.output)
    print(json.dumps({key: result[key] for key in ("status", "scenario_count", "view_count", "screenshot_dimensions")}))
