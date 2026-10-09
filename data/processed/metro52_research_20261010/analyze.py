"""Aggregate completed-day ledgers without model calls or passenger replay."""
import sys, json, gzip, csv, math, hashlib, gc
from pathlib import Path
from collections import Counter, defaultdict
from bisect import bisect_right
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from metro50_economics import economy_comparison
from interfaces.streamlit_dispatcher.adapter import NAMES

OUT = Path(__file__).resolve().parent
RUN = ROOT/'data/processed/metro52_full_day_20261009'
END = 72000
def read(p):
    raw=p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(hist):
    n=sum(hist.values())
    def q(p):
        rank=math.ceil(n*p); seen=0
        for x,k in sorted(hist.items()):
            seen+=k
            if seen>=rank:return x
    return dict(count=n,mean=sum(x*k for x,k in hist.items())/n if n else None,
                p50=q(.5) if n else None,p95=q(.95) if n else None,
                p99=q(.99) if n else None,maximum=max(hist,default=None))
assert stats(Counter({0:95,100:5}))['p95']==0
assert stats(Counter({0:95,100:5}))['p99']==100
def save(name,value): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvsave(name,rows):
    if not rows:return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def station(s):return NAMES[int(s.split('_')[-1])-1]
def direction(d):return 'к Девяткино' if d==1 else 'к Проспекту Ветеранов'

common_path=RUN/'balanced/common_scenario.json.gz'
comparison=read(RUN/'review/comparison.json')
assert comparison['paired_inputs_passed']
assert all(x['checkpoint_verified_against_recompiled_runner'] for x in comparison['audits'].values())
sources=[common_path,RUN/'review/comparison.json',RUN/'forecast_review/report.json',RUN/'incidents/result.json']
for arm in ['reference','balanced']:
    result=read(RUN/arm/'result.json');rd=RUN/arm/result['run']['folder']
    sources.extend([RUN/arm/'result.json',rd/'passenger_checkpoint.json.gz',rd/'motion.json'])
hashes={str(p):sha(p) for p in sources}
common=read(common_path); groups=common['groups']; by={g['group_id']:g for g in groups}
origin=datetime.fromisoformat(common['data']['period_start'])
clock=lambda second:(origin+timedelta(seconds=second)).strftime('%H:%M:%S')
results={};ledgers={};histograms={};station_rows=[];period_rows=[];bin_rows=[];queue_rows=[];long_rows=[];service_rows=[]
for arm in ['reference','balanced']:
    print('reading',arm,flush=True)
    result=read(RUN/arm/'result.json');metrics=result['metrics'];rd=RUN/arm/result['run']['folder']
    state=read(rd/'passenger_checkpoint.json.gz');motion=read(rd/'motion.json')
    assert state['second']==END
    motion_by_id={e['event_id']:e for e in motion['events']}
    boards=state['board_events']; hist=Counter();local=defaultdict(Counter); periods=defaultdict(Counter);quarters=defaultdict(Counter)
    arrivals=defaultdict(list)
    for e in motion['events']:
        if e['event_type']=='arrival':arrivals[e['train_id'],e['location']].append(e['second'])
    for a in arrivals.values():a.sort()
    trip_hist=Counter();ride_hist=Counter();gate_hist=Counter();mature_hist=Counter();complete_count=0
    for row in boards:
        g=by[row['group_id']];w=row['second']-g['platform_second'];n=row['count']
        assert w==row['wait_seconds'] and row['second']<END
        dr=1 if row['destination']>g['origin'] else -1
        hist[w]+=n;local[g['origin'],dr][w]+=n;gate_hist[row['second']-g['turnstile_second']]+=n
        if g['platform_second']<END-600:mature_hist[w]+=n
        hour=(origin+timedelta(seconds=g['platform_second'])).hour
        period='Утренний пик 08–10' if 8<=hour<10 else 'Вечерний пик 17–19' if 17<=hour<19 else 'Вне пика'
        periods[period][w]+=n;quarters[g['platform_second']//900*900][w]+=n
        event=motion_by_id[row['motion_event_id']]
        a=arrivals[event['train_id'],row['destination']];idx=bisect_right(a,row['second'])
        assert idx<len(a)
        at=a[idx]
        if at<END:
            complete_count+=n;ride_hist[at-row['second']]+=n;trip_hist[at-g['turnstile_second']]+=n
        if w>=300:
            long_rows.append(dict(arm=arm,station=station(g['origin']),direction=direction(dr),destination=station(row['destination']),platform_time=clock(g['platform_second']),board_time=clock(row['second']),wait_seconds=w,count=n,group_id=g['group_id']))
    entered=sum(g['count'] for g in groups if g['turnstile_second']<END)
    platform=sum(g['count'] for g in groups if g['platform_second']<END)
    queued=sum(r['count'] for q in state['queues'].values() for r in q)
    last_load={}
    for row in state['load_events']:last_load[row['train_id'],row['car_id']]=row['load']
    onboard=sum(last_load.values());boarded=sum(hist.values())
    assert boarded+queued==platform and entered==complete_count+onboard+queued+entered-platform
    if True:
        assert (boarded,queued,onboard,complete_count)==(state['boarded'],state['queued'],state['onboard_count'],state['completed'])
        residual=sum((END-r['platform_second'])*r['count'] for q in state['queues'].values() for r in q)
        assert sum(w*n for w,n in hist.items())+residual==state['waiting_passenger_seconds']
    queuehist=Counter()
    for queue in state['queues'].values():
        for row in queue:queuehist[END-row['platform_second']]+=row['count']
    for (sid,dr),h in local.items():
        qrows=[x for q in state['queues'].values() for x in q if x['origin']==sid and x['direction']==dr]
        station_rows.append(dict(arm=arm,station=station(sid),direction=direction(dr),**stats(h),still_waiting=sum(x['count'] for x in qrows),longest_current_wait=max((END-x['platform_second'] for x in qrows),default=0)))
    for (label,h) in periods.items():period_rows.append(dict(arm=arm,period=label,**stats(h)))
    for lo,hi,label in [(0,60,'0–1 мин'),(60,120,'1–2 мин'),(120,180,'2–3 мин'),(180,300,'3–5 мин'),(300,480,'5–8 мин'),(480,10**9,'8+ мин')]:
        n=sum(n for w,n in hist.items() if lo<=w<hi);bin_rows.append(dict(arm=arm,wait=label,count=n,percent=n/boarded*100))
    for sid in sorted(set(g['origin'] for g in groups)):
        for dr in [-1,1]:
            qr=[r for q in state['queues'].values() for r in q if r['origin']==sid and r['direction']==dr]
            if qr:queue_rows.append(dict(arm=arm,station=station(sid),direction=direction(dr),count=sum(r['count'] for r in qr),mean_age_seconds=sum((END-r['platform_second'])*r['count'] for r in qr)/sum(r['count'] for r in qr),max_age_seconds=max(END-r['platform_second'] for r in qr)))
    service=metrics['local_boarding_service']
    assert abs(stats(hist)['mean']-metrics['cohort']['mean_wait_seconds'])<1e-9
    assert boarded==complete_count==758936 and queued==onboard==0
    for c in service['cohorts']:
        service_rows.append(dict(arm=arm,station=station(c['station_id']),direction=direction(c['direction']),start=c['interval_start'],passengers=c['passenger_count'],missed_first=c['missed_first_count'],missed_second=c['missed_second_count'],first_percent=100*c['missed_first_share'],second_percent=100*c['missed_second_share'],status=c['status']))
    histograms[arm]=dict(wait=hist,quarters=quarters,queue=[(r['second'],r['queued']) for r in state['queue_events']])
    ledgers[arm]=metrics['actual_per_train']
    peakload=Counter(r['load'] for r in state['load_events'] if r.get('event_type')=='departure' and (8<=(origin+timedelta(seconds=r['second'])).hour<10 or 17<=(origin+timedelta(seconds=r['second'])).hour<19))
    results[arm]=dict(entered=entered,boarded=boarded,completed=complete_count,queued=queued,approaching=entered-platform,onboard=onboard,waiting_boarded=stats(hist),gate_to_board=stats(gate_hist),waiting_mature=stats(mature_hist),mature_cohort_end=END-600,mature_passengers=sum(g['count'] for g in groups if g['platform_second']<END-600),current_queue_age=stats(queuehist),completed_trip=stats(trip_hist),completed_ride=stats(ride_hist),max_queue=max(r['queued'] for r in state['queue_events']),max_car_load=max(r['load'] for r in state['load_events']),peak_car_departure_load=stats(peakload),service=service['metrics'],failed_service_cohorts=service['failed_cohort_count'],service_scope_end=service['cohort_end_second'],train_km=sum(x['km'] for x in ledgers[arm]))
    assert results[arm]['mature_passengers']==results[arm]['waiting_mature']['count']
    results[arm]['headways']={k:v for k,v in metrics['headways'].items() if not isinstance(v,(list,dict))}
    print('metrics',arm,{k:results[arm][k] for k in ['boarded','completed','queued','train_km']},flush=True)
    del state,motion,motion_by_id
    gc.collect()
economy=comparison['economics']['balanced']
decisions=[read(p) for p in sorted((RUN/'balanced/controller').glob('decision_*.json')) if int(p.stem.split('_')[-1])<=END]
actions=[dict(time=d['decision_time'],action=d.get('applied_candidate_id'),pipeline_seconds=d.get('pipeline_seconds'),status=d['selection']['status']) for d in decisions]
summary=dict(forecast=read(RUN/'forecast_review/report.json'),explanation_summary=comparison['explanation_summary'],audit={k:{'passed':v['passed'],'failed_checks':v['failed_checks']} for k,v in comparison['audits'].items()},goal_achieved=comparison['goal_achieved'],cutoff_second=END,cutoff_time=(origin+timedelta(seconds=END)).isoformat(),steps=len(decisions),full_day=True,passengers=results,economy=economy,actions=actions,source_sha256=hashes,baseline='fixed synthetic calendar/FIFO reference',cash_saving_rub=None,release_authorized=False)
save('summary.json',summary)
for name,rows in [('station_waits.csv',station_rows),('period_waits.csv',period_rows),('wait_distribution.csv',bin_rows),('current_queues.csv',queue_rows),('long_waits.csv',sorted(long_rows,key=lambda x:x['wait_seconds'],reverse=True)),('local_service.csv',service_rows),('decisions.csv',actions)]:csvsave(name,rows)
assert all(sha(Path(p))==h for p,h in hashes.items())
save('verification.json',dict(source_hashes_unchanged=True,passenger_balance_passed=True,independent_final_audit_checkpoint_validated=True,mature_cohort_complete_both=True,observation_end_second=END,model_calls=0,passenger_replays=0))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
colors={'reference':'#617386','balanced':'#d63a46'}
labels={'reference':'Исходный график','balanced':'Ансамбль + оптимизатор'}
fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
for arm in results:
    h=histograms[arm]['wait'];n=sum(h.values());xs=sorted(h);acc=0;ys=[]
    for x in xs:acc+=h[x];ys.append(100*acc/n)
    axes[0,0].plot([x/60 for x in xs],ys,label=labels[arm],color=colors[arm],linestyle='--' if arm=='reference' else '-')
    q=histograms[arm]['quarters'];axes[0,1].plot([6+t/3600 for t in q],[stats(q[t])['mean'] for t in q],color=colors[arm],label=labels[arm],linestyle='--' if arm=='reference' else '-')
    rows=sorted([r for r in station_rows if r['arm']==arm],key=lambda r:r['p95'],reverse=True)
    if arm=='balanced':
        rows=rows[:8];axes[1,0].barh([r['station']+' '+('↓' if r['direction'].startswith('к Проспекту') else '↑') for r in rows][::-1],[r['p95']/60 for r in rows][::-1],color=colors[arm])
    pair=histograms[arm]['queue'];axes[1,1].plot([6+t/3600 for t,n in pair],[n for t,n in pair],color=colors[arm],alpha=.8,linestyle='--' if arm=='reference' else '-',label=labels[arm])
axes[0,0].set(title='Распределение ожидания всех пассажиров',xlabel='Ожидание на платформе, мин',ylabel='Доля пассажиров, %',xlim=(0,8),ylim=(0,101));axes[0,0].legend(fontsize=9)
axes[0,1].set(title='Среднее ожидание по времени прихода',xlabel='Часы моделирования',ylabel='Секунды');axes[0,1].axvspan(8,10,color='#ffc857',alpha=.16);axes[0,1].axvspan(17,19,color='#ffc857',alpha=.16)
axes[1,0].set(title='Где выше P95 ожидания: новый вариант',xlabel='Минуты');axes[1,1].set(title='Суммарная очередь на платформах',xlabel='Часы моделирования',ylabel='Пассажиры')
for ax in axes.flat:ax.grid(alpha=.2,axis='x')
fig.suptitle('Красная линия • 30.09.2026 • полный пассажирский день',fontsize=17)
fig.savefig(OUT/'passenger_charts.png',dpi=160);fig.savefig(OUT/'passenger_charts.pdf');plt.close(fig)
print('DONE',str(OUT),flush=True)
