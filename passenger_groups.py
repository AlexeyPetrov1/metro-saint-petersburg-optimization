"""Grouped extension of the passenger event stream with typed car capacities."""

from __future__ import annotations

import copy
import heapq
import json
from collections import Counter, deque
from decimal import Decimal

from economics import number, ratio
from passengers import PassengerSimulation
from simulation import digest, fields_only, identifier, indexed, integer


MODEL_VERSION = "passenger-groups-v1"


def allocate_preferred(count,loads,capacities,weights):
    """Engineering sensitivity: equalize load/weight, with physical caps."""
    allocate_cars(0,loads,capacities)
    integer(count,"boarding count")
    if len(weights)!=len(loads): raise ValueError("preference dimensions differ")
    for weight in weights: integer(weight,"car preference",1)
    count=min(count,sum(c-l for c,l in zip(capacities,loads)))
    low,high=0,max(capacities)
    def additions(level): return [max(0,min(c,w*level)-l) for l,c,w in zip(loads,capacities,weights)]
    while low<high:
        level=(low+high+1)//2
        if sum(additions(level))<=count: low=level
        else: high=level-1
    result=additions(low); remaining=count-sum(result)
    heap=[((l+a)/w,i) for i,(l,a,w,c) in enumerate(zip(loads,result,weights,capacities)) if l+a<c]
    heapq.heapify(heap)
    while remaining:
        _,i=heapq.heappop(heap); result[i]+=1; remaining-=1
        if loads[i]+result[i]<capacities[i]: heapq.heappush(heap,((loads[i]+result[i])/weights[i],i))
    return result


def allocate_cars(count, loads, capacities):
    """Batch equivalent of repeatedly choosing the least-loaded available car."""
    integer(count, "boarding count")
    if len(loads) != len(capacities) or not loads:
        raise ValueError("car load and capacity dimensions differ")
    for load, cap in zip(loads, capacities):
        integer(load, "car load"); integer(cap, "car capacity", 1)
        if load > cap:
            raise ValueError("car load exceeds boarding capacity")
    count = min(count, sum(c-l for l,c in zip(loads, capacities)))
    if not count: return [0]*len(loads)
    if count==sum(c-l for l,c in zip(loads,capacities)):
        return [c-l for l,c in zip(loads,capacities)]
    active=[i for i,c in enumerate(capacities) if loads[i]<c]
    # Most batches follow an already balanced boarding batch. Avoid a binary
    # search when equalization cannot hit a car's physical admission limit.
    if max(loads[i] for i in active)-min(loads[i] for i in active)<=1:
        total=sum(loads[i] for i in active)+count
        level,remainder=divmod(total,len(active))
        if level+bool(remainder)<=min(capacities[i] for i in active):
            additions=[max(0,level-l) if i in active else 0 for i,l in enumerate(loads)]
            remainder=count-sum(additions)
            for i in active:
                if remainder and loads[i]+additions[i]==level:
                    additions[i]+=1; remainder-=1
            return additions
    low, high = 0, max(capacities)
    while low < high:
        level = (low+high+1)//2
        needed = sum(max(0, min(c,level)-l) for l,c in zip(loads, capacities))
        if needed <= count: low = level
        else: high = level-1
    additions = [max(0,min(c,low)-l) for l,c in zip(loads, capacities)]
    remainder = count-sum(additions)
    for i, (load, cap) in enumerate(zip(loads, capacities)):
        if remainder and load+additions[i] == low and cap > low:
            additions[i] += 1
            remainder -= 1
    if remainder:
        raise ValueError("car allocation failed to conserve counts")
    return additions


class GroupedPassengerSimulation:
    def __init__(self, movement, plan_id, groups, profiles, *, car_preference=None,
                 boarding_seed=None, boarding_capacity_per_car=None):
        if not isinstance(movement,dict) or not isinstance(profiles,dict) or not isinstance(groups,list):
            raise ValueError("movement and profiles must be objects; groups must be a list")
        trains=indexed(movement.get("trains"),"train_id")
        self.profiles = copy.deepcopy(profiles)
        self.car_preference=copy.deepcopy(car_preference)
        if boarding_seed is not None: integer(boarding_seed,"boarding seed")
        if boarding_seed is not None and car_preference is not None:
            raise ValueError("boarding seed and explicit car preference are mutually exclusive")
        if boarding_capacity_per_car is not None: integer(boarding_capacity_per_car,"boarding capacity",1)
        self.boarding_seed=boarding_seed
        self.boarding_capacity_per_car=boarding_capacity_per_car
        if set(profiles) != set(trains):
            raise ValueError("car profiles must cover the supplied fleet exactly")
        total_caps = []
        for train in movement["trains"]:
            integer(train.get("cars"),"train cars",1)
            cars = profiles[train["train_id"]]
            if not isinstance(cars,list) or len(cars) != train["cars"]:
                raise ValueError("typed car profile differs from train formation")
            ids = set()
            for car in cars:
                fields_only(car, ("car_id", "car_type", "max_capacity", "normal_threshold", "source"))
                for key in ("car_id", "car_type", "source"): identifier(car[key], key)
                for key in ("max_capacity", "normal_threshold"): integer(car[key],key,1)
                if car["car_id"] in ids or car["normal_threshold"] > car["max_capacity"]:
                    raise ValueError("duplicate car or threshold above capacity")
                if boarding_capacity_per_car is not None and boarding_capacity_per_car>car["max_capacity"]:
                    raise ValueError("boarding capacity exceeds physical car capacity")
                ids.add(car["car_id"])
            total_caps.append({"train_id":train["train_id"], "max_capacity":sum(c["max_capacity"] for c in cars),
                "normal_threshold":sum(c["normal_threshold"] for c in cars), "source":"typed grouped car profile"})
        # Reuse the existing passenger compiler and its audited movement result.
        base = PassengerSimulation(movement, plan_id, {"schema_version":"1.0", "source":"grouped passenger extension",
                                                       "arrivals":[], "capacities":total_caps})
        self.motion, self.tracks = base.motion, base.tracks
        self.duration = self.motion["duration_seconds"]
        self.groups = copy.deepcopy(groups)
        self.station_ids = [s["station_id"] for s in movement["stations"]]
        self.positions = {s:i for i,s in enumerate(self.station_ids)}
        self.boarding_capacities={t:[c["max_capacity"] if boarding_capacity_per_car is None else
            boarding_capacity_per_car for c in cars] for t,cars in self.profiles.items()}
        # Keyed platform preferences have no mutable random state, so replay and
        # future-plan branches share the same mild crowd concentration.
        self._boarding_preferences={} if boarding_seed is None else {
            (s,d,n):[90+int(digest((boarding_seed,s,d,i))[:8],16)%21 for i in range(n)]
            for s in self.station_ids for d in (-1,1) for n in {len(c) for c in self.profiles.values()}}
        self.by_id, events = {}, []
        for group in self.groups:
            fields_only(group,("group_id","turnstile_second","platform_second","origin","count","destinations"),
                        ("source_id","source_interval_start","source_interval_count"))
            identifier(group["group_id"],"group_id")
            identifier(group["origin"],"origin")
            integer(group["turnstile_second"],"turnstile_second")
            if not isinstance(group["destinations"],list): raise ValueError("destinations must be a list")
        for group in sorted(self.groups,key=lambda g:(g["turnstile_second"],g["group_id"])):
            name=identifier(group["group_id"],"group_id")
            if name in self.by_id: raise ValueError("duplicate input group")
            gate=integer(group["turnstile_second"],"turnstile_second")
            platform=integer(group["platform_second"],"platform_second")
            integer(group["count"],"group count")
            if platform < gate or gate >= self.duration or group["origin"] not in self.positions:
                raise ValueError("invalid entry, platform time or origin")
            total, seen = 0,set()
            for dest in group["destinations"]:
                fields_only(dest,("station_id","count"),("direction",))
                identifier(dest["station_id"],"destination station")
                s=dest["station_id"]
                integer(dest["count"],"destination count",1)
                if s not in self.positions or s == group["origin"] or s in seen:
                    raise ValueError("destination must be a distinct line station")
                direction=1 if self.positions[s]>self.positions[group["origin"]] else -1
                if "direction" in dest and integer(dest["direction"],"direction",-1) != direction:
                    raise ValueError("OD direction contradicts ordered stations")
                dest["direction"]=direction
                total+=dest["count"]; seen.add(s)
            if total != group["count"]: raise ValueError("OD quantities differ from source group")
            self.by_id[name]=group
            events += [(gate,0,name,{"kind":"gate","second":gate,"group_id":name}),
                       (platform,1,name,{"kind":"platform","second":platform,"group_id":name})]
        for i,e in enumerate(base.stream):
            priority={"arrival":2,"departure":4}.get(e["event_type"],3)
            events.append((e["second"],priority,f"{i:012}",e))
        for second in range(0,self.duration,60):
            events.append((second,5,"",{"kind":"sample","second":second}))
        self.stream=[e for _,_,_,e in sorted(events)]
        self.context_id=digest({"model_version":MODEL_VERSION,"motion":self.motion,
                                "groups":list(self.by_id.values()),"profiles":self.profiles})
        if car_preference is not None:
            for cars in self.profiles.values(): allocate_preferred(0,[0]*len(cars),[c["max_capacity"] for c in cars],car_preference)
            self.context_id=digest({"base":self.context_id,"car_preference":car_preference})
        if boarding_seed is not None or boarding_capacity_per_car is not None:
            self.context_id=digest({"base":self.context_id,"boarding_seed":boarding_seed,
                "boarding_capacity_per_car":boarding_capacity_per_car,"boarding_preferences_version":"station-v1"})
        self.motion_by_id={e["event_id"]:e for e in base.stream}

    def _initial(self):
        return {"model_version":MODEL_VERSION,"context_id":self.context_id,"second":0,"next_event_index":0,
            "entered":0,"platform_arrived":0,"boarded":0,"completed":0,"approaching":0,
            "queued":0,"onboard_count":0,"waiting_passenger_seconds":0,"wait_histogram":{},
            "queues":{},"onboard":{t:[[] for _ in cars] for t,cars in self.profiles.items()},
            "loads":{t:[0]*len(cars) for t,cars in self.profiles.items()},"queue_counts":{},
            "max_queue":0,"max_platform_direction_queue":0,"max_queue_by_direction":{},"max_car_load":0,
            "normal_threshold_exceedances":0,"denied_boarding_attempts":0,
            "queue_samples":[],"queue_events":[],"load_events":[],"board_events":[]}

    def branch_checkpoint(self, checkpoint, source):
        """Change future plans/demand only after verifying the shared past."""
        start = checkpoint["second"]
        source._validate_checkpoint(checkpoint,start)
        if self.boarding_seed!=source.boarding_seed or self.boarding_capacity_per_car!=source.boarding_capacity_per_car:
            raise ValueError("branch changes boarding seed or boarding capacity")
        if self.profiles != source.profiles or self.car_preference != source.car_preference or self.motion["initial_state"] != source.motion["initial_state"]:
            raise ValueError("branch changes the initial fleet or past capacity")
        for key,group in source.by_id.items():
            if group["turnstile_second"] < start and self.by_id.get(key) != group:
                raise ValueError("branch rewrites past passenger arrivals")
        previous=[e for e in source.stream if e["second"]<start]
        current=[e for e in self.stream if e["second"]<start]
        if previous != current:
            raise ValueError("branch rewrites past movement or demand")
        state=json.loads(json.dumps(checkpoint,ensure_ascii=False))
        state.update(context_id=self.context_id,next_event_index=len(current))
        state["sha256"]=PassengerSimulation._seal(state)
        self._validate_checkpoint(state,start)
        return state

    def _queue_age(self,state):
        return sum(g["count"]*(state["second"]-g["platform_second"]) for queue in state["queues"].values() for g in queue)

    def _validate_checkpoint(self,state,start):
        """Check stock and ledger consistency; a checksum is not authentication."""
        fields_only(state,(*self._initial(),"sha256"))
        if (state["sha256"]!=PassengerSimulation._seal(state) or state["context_id"]!=self.context_id
                or state["model_version"]!=MODEL_VERSION or integer(state["second"],"checkpoint second")!=start or not start):
            raise ValueError("invalid checkpoint or nonconsecutive window")
        if integer(state["next_event_index"],"event cursor")!=sum(e["second"]<start for e in self.stream):
            raise ValueError("checkpoint event cursor mismatch")
        for key,value in self._initial().items():
            if type(value) is int: integer(state[key],key)
            elif isinstance(value,dict) and not isinstance(state[key],dict): raise ValueError(f"{key} must be an object")
            elif isinstance(value,list) and not isinstance(state[key],list): raise ValueError(f"{key} must be a list")
        valid_keys={f"{s}:{d}" for s in self.station_ids for d in (1,-1)}
        for key in ("queues","queue_counts","max_queue_by_direction"):
            if set(state[key])-valid_keys: raise ValueError("unknown platform direction in checkpoint")
        for key in ("loads","onboard"):
            if set(state[key])!=set(self.profiles): raise ValueError("checkpoint fleet differs from movement")

        by_group,by_od,queue_counts=Counter(),Counter(),Counter()
        def cohort(row,boarded=False):
            fields_only(row,("group_id","origin","destination","direction","turnstile_second","platform_second","count")
                        + (("boarded_second",) if boarded else ()))
            parent=self.by_id.get(row["group_id"])
            if not parent: raise ValueError("checkpoint contains an unknown group")
            for key in ("origin","turnstile_second","platform_second"):
                if type(row[key]) is not type(parent[key]) or row[key]!=parent[key]:
                    raise ValueError("checkpoint changed group identity or arrival")
            dest=next((d for d in parent["destinations"] if d["station_id"]==row["destination"]),None)
            if not dest or integer(row["direction"],"direction",-1)!=dest["direction"]:
                raise ValueError("checkpoint changed destination or direction")
            count=integer(row["count"],"cohort count",1)
            if parent["platform_second"]>=start: raise ValueError("cohort precedes its platform arrival")
            by_od[row["group_id"],row["destination"]]+=count
            if by_od[row["group_id"],row["destination"]]>dest["count"]:
                raise ValueError("checkpoint duplicates a destination cohort")
            return parent,count

        for key,rows in state["queues"].items():
            if not isinstance(rows,list): raise ValueError("queue cohorts must be a list")
            previous=None; seen=set()
            for row in rows:
                parent,count=cohort(row)
                if key!=f"{row['origin']}:{row['direction']}": raise ValueError("cohort is in the wrong queue")
                order=(row["platform_second"],row["group_id"],self.positions[row["destination"]])
                identity=(row["group_id"],row["destination"])
                if identity in seen or previous is not None and order<previous: raise ValueError("queue duplicates or violates FIFO")
                previous=order; seen.add(identity); by_group[row["group_id"]]+=count; queue_counts[key]+=count
        for key in set(queue_counts)|set(state["queue_counts"]):
            if integer(state["queue_counts"].get(key,0),"queue count")!=queue_counts[key]:
                raise ValueError("queue counters differ from cohorts")
        if state["queued"]!=sum(queue_counts.values()): raise ValueError("queued stock differs from cohorts")

        board_hist,boarded_by_group,board_at,boarded_by_destination=Counter(),Counter(),Counter(),Counter()
        previous=-1
        for row in state["board_events"]:
            fields_only(row,("second","wait_seconds","count","group_id","motion_event_id"),("destination",))
            second=integer(row["second"],"boarding second"); count=integer(row["count"],"boarding count",1)
            parent=self.by_id.get(row["group_id"]); event=self.motion_by_id.get(row["motion_event_id"])
            if (not parent or not event or event["event_type"]!="departure" or not event["passenger_service"]
                    or second!=event["second"] or not previous<=second<start
                    or self.tracks[event["location"]]["from_station"]!=parent["origin"]
                    or integer(row["wait_seconds"],"waiting age")!=second-parent["platform_second"]):
                raise ValueError("invalid boarding history")
            if self.boarding_seed is not None or self.boarding_capacity_per_car is not None or "destination" in row:
                destination=row.get("destination")
                dest=next((d for d in parent["destinations"] if d["station_id"]==destination),None)
                boarded_by_destination[row["group_id"],destination]+=count
                if (not dest or boarded_by_destination[row["group_id"],destination]>dest["count"]
                        or destination not in {self.tracks[t]["to_station"] for t in event["remaining_route"]}):
                    raise ValueError("boarding destination contradicts cohort or passenger route")
            previous=second; board_hist[str(row["wait_seconds"])]+=count
            boarded_by_group[row["group_id"]]+=count
            board_at[row["group_id"],second,event["train_id"]]+=count
        for wait,count in state["wait_histogram"].items():
            if not isinstance(wait,str) or not wait.isdecimal() or str(int(wait))!=wait:
                raise ValueError("invalid waiting histogram key")
            integer(count,"waiting histogram count",1)
        if dict(board_hist)!=state["wait_histogram"] or sum(board_hist.values())!=state["boarded"]:
            raise ValueError("boarding histogram differs from ledger")

        onboard=0; onboard_at=Counter()
        for train,profile in self.profiles.items():
            cars,loads=state["onboard"][train],state["loads"][train]
            if not isinstance(cars,list) or not isinstance(loads,list) or len(cars)!=len(profile) or len(loads)!=len(profile):
                raise ValueError("checkpoint car dimensions differ from formation")
            for rows,load,car,cap in zip(cars,loads,profile,self.boarding_capacities[train]):
                if not isinstance(rows,list): raise ValueError("onboard cohorts must be a list")
                total=0
                for row in rows:
                    parent,count=cohort(row,True); board=integer(row["boarded_second"],"boarding second")
                    if not parent["platform_second"]<=board<start or (row["group_id"],board,train) not in board_at:
                        raise ValueError("onboard cohort lacks its boarding event")
                    onboard_at[row["group_id"],board,train]+=count; total+=count
                if integer(load,"car load")!=total or total>cap:
                    raise ValueError("car load differs from cohorts or exceeds capacity")
                onboard+=total
        if any(n>board_at[key] for key,n in onboard_at.items()) or onboard!=state["onboard_count"]:
            raise ValueError("onboard stock differs from boarding history")
        entered=sum(g["count"] for g in self.groups if g["turnstile_second"]<start)
        platform=sum(g["count"] for g in self.groups if g["platform_second"]<start)
        expected={"entered":entered,"platform_arrived":platform,"approaching":entered-platform,
                  "queued":platform-state["boarded"],"completed":state["boarded"]-onboard}
        if any(state[k]!=v for k,v in expected.items()): raise ValueError("checkpoint passenger balance differs from input")
        if any(by_group[g["group_id"]]+boarded_by_group[g["group_id"]]!=
               (g["count"] if g["platform_second"]<start else 0) for g in self.groups):
            raise ValueError("checkpoint source group balance failed")
        wait=sum(int(w)*n for w,n in board_hist.items())+self._queue_age(state)
        if state["waiting_passenger_seconds"]!=wait: raise ValueError("checkpoint waiting-time conservation failed")
        self._validate_history(state,start)

    def _validate_history(self,state,start):
        last_loads={t:[0]*len(cars) for t,cars in self.profiles.items()}
        expected=[(e,car,i) for e in self.stream if e["second"]<start and e["kind"]=="motion"
                  and (e["event_type"]=="arrival" or e["event_type"]=="departure" and e["passenger_service"])
                  for i,car in enumerate(self.profiles[e["train_id"]])]
        if len(state["load_events"])!=len(expected): raise ValueError("incomplete car load history")
        for row,(event,car,i) in zip(state["load_events"],expected):
            fields_only(row,("second","train_id","car_id","car_type","load","normal_threshold","max_capacity",
                             "threshold_exceeded","motion_event_id","event_type"))
            values={"second":event["second"],"train_id":event["train_id"],"motion_event_id":event["event_id"],
                    "event_type":event["event_type"],**{k:car[k] for k in ("car_id","car_type","normal_threshold","max_capacity")}}
            load=integer(row["load"],"historical car load")
            if (any(type(row[k]) is not type(v) or row[k]!=v for k,v in values.items())
                    or load>self.boarding_capacities[event["train_id"]][i]
                    or type(row["threshold_exceeded"]) is not bool or row["threshold_exceeded"]!=(load>car["normal_threshold"])):
                raise ValueError("car history contradicts movement or profile")
            last_loads[event["train_id"]][i]=load
        if last_loads!=state["loads"]: raise ValueError("car stock differs from last observation")
        if state["max_car_load"]!=max((r["load"] for r in state["load_events"]),default=0): raise ValueError("invalid car peak")
        if state["normal_threshold_exceedances"]!=sum(r["threshold_exceeded"] for r in state["load_events"]):
            raise ValueError("invalid threshold observation count")
        for row in state["queue_events"]:
            fields_only(row,("second","queued","max_platform_direction_queue"))
            if integer(row["second"],"queue history second")>=start: raise ValueError("queue history extends beyond checkpoint")
            if integer(row["max_platform_direction_queue"],"direction queue")>integer(row["queued"],"historical queue"):
                raise ValueError("direction queue exceeds total queue")
        for key,field in (("max_queue","queued"),("max_platform_direction_queue","max_platform_direction_queue")):
            if state[key]!=max((r[field] for r in state["queue_events"]),default=0): raise ValueError("invalid queue peak")
        for key,peak in state["max_queue_by_direction"].items():
            if not state["queue_counts"].get(key,0)<=integer(peak,"direction peak")<=state["max_platform_direction_queue"]:
                raise ValueError("invalid direction peak")
        samples=set()
        for row in state["queue_samples"]:
            fields_only(row,("second","station_id","direction","queued"))
            second=integer(row["second"],"sample second"); direction=integer(row["direction"],"sample direction",-1)
            key=(second,row["station_id"],direction)
            integer(row["queued"],"sample queue")
            if second>=start or second%60 or row["station_id"] not in self.positions or direction not in (-1,1) or key in samples:
                raise ValueError("invalid or repeated queue sample")
            samples.add(key)
        if len(samples)!=len(range(0,start,60))*len(self.station_ids)*2: raise ValueError("incomplete queue samples")

    def _load_log(self,state,train,event):
        for i,(load,car) in enumerate(zip(state["loads"][train],self.profiles[train])):
            exceeded=load>car["normal_threshold"]
            state["max_car_load"]=max(state["max_car_load"],load)
            state["normal_threshold_exceedances"]+=int(exceeded)
            state["load_events"].append({"second":state["second"],"train_id":train,"car_id":car["car_id"],
                "car_type":car["car_type"],"load":load,"normal_threshold":car["normal_threshold"],
                "max_capacity":car["max_capacity"],"threshold_exceeded":exceeded,
                "motion_event_id":event["event_id"],"event_type":event["event_type"]})

    def _queue_log(self,state):
        state["queue_events"].append({"second":state["second"],"queued":state["queued"],
            "max_platform_direction_queue":max(state["queue_counts"].values(),default=0)})

    def _process(self,state,e):
        if e["kind"] == "gate":
            count=self.by_id[e["group_id"]]["count"]
            state["entered"]+=count; state["approaching"]+=count
        elif e["kind"] == "platform":
            parent=self.by_id[e["group_id"]]
            state["platform_arrived"]+=parent["count"]; state["approaching"]-=parent["count"]
            for dest in sorted(parent["destinations"],key=lambda d:self.positions[d["station_id"]]):
                key=f"{parent['origin']}:{dest['direction']}"
                row={"group_id":parent["group_id"],"origin":parent["origin"],"destination":dest["station_id"],
                    "direction":dest["direction"],"turnstile_second":parent["turnstile_second"],
                    "platform_second":parent["platform_second"],"count":dest["count"]}
                state["queues"].setdefault(key,deque()).append(row)
                state["queued"]+=row["count"]
                state["queue_counts"][key]=state["queue_counts"].get(key,0)+row["count"]
                state["max_queue_by_direction"][key]=max(state["max_queue_by_direction"].get(key,0),state["queue_counts"][key])
                state["max_platform_direction_queue"]=max(state["max_platform_direction_queue"],state["queue_counts"][key])
            state["max_queue"]=max(state["max_queue"],state["queued"])
            self._queue_log(state)
        elif e["kind"] == "sample":
            for station in self.station_ids:
                for direction in (1,-1):
                    state["queue_samples"].append({"second":e["second"],"station_id":station,"direction":direction,
                                                  "queued":state["queue_counts"].get(f"{station}:{direction}",0)})
        elif e["event_type"] == "arrival":
            train=e["train_id"]
            for i,car in enumerate(state["onboard"][train]):
                kept=[]
                for g in car:
                    if g["destination"] == e["location"]:
                        state["completed"]+=g["count"]; state["onboard_count"]-=g["count"]
                        state["loads"][train][i]-=g["count"]
                    else: kept.append(g)
                state["onboard"][train][i]=kept
            self._load_log(state,train,e)
        elif e["event_type"] == "departure":
            if not e["passenger_service"]: return
            train=e["train_id"]; track=self.tracks[e["location"]]
            direction=1 if self.positions[track["to_station"]]>self.positions[track["from_station"]] else -1
            key=f"{track['from_station']}:{direction}"
            queue=state["queues"].get(key,deque())
            destinations={self.tracks[t]["to_station"] for t in e["remaining_route"]}
            caps=self.boarding_capacities[train]
            preference=(self.car_preference if self.boarding_seed is None else
                self._boarding_preferences[track["from_station"],direction,len(caps)])
            remaining=[]
            while queue:
                group=queue.popleft()
                if group["destination"] not in destinations:
                    remaining.append(group); continue
                additions=(allocate_cars(group["count"],state["loads"][train],caps) if preference is None else
                    allocate_preferred(group["count"],state["loads"][train],caps,preference))
                boarded=sum(additions)
                if boarded:
                    wait=state["second"]-group["platform_second"]
                    state["wait_histogram"][str(wait)]=state["wait_histogram"].get(str(wait),0)+boarded
                    board={"second":state["second"],"wait_seconds":wait,"count":boarded,
                           "group_id":group["group_id"],"motion_event_id":e["event_id"]}
                    if self.boarding_seed is not None or self.boarding_capacity_per_car is not None:
                        board["destination"]=group["destination"]
                    state["board_events"].append(board)
                    for i,count in enumerate(additions):
                        if count:
                            state["onboard"][train][i].append({**group,"count":count,"boarded_second":state["second"]})
                            state["loads"][train][i]+=count
                    state["boarded"]+=boarded; state["onboard_count"]+=boarded
                    state["queued"]-=boarded; state["queue_counts"][key]-=boarded
                    group["count"]-=boarded
                if group["count"]:
                    remaining.append(group)
            state["queues"][key]=deque(remaining)
            state["denied_boarding_attempts"]+=sum(g["count"] for g in remaining if g["destination"] in destinations)
            self._queue_log(state)
            self._load_log(state,train,e)

    def run(self,*,start_second=0,end_second=None,checkpoint=None,deadline=None,cancel_event=None):
        from time import monotonic
        def check_deadline():
            if cancel_event is not None and cancel_event.is_set() or deadline is not None and monotonic()>=deadline:
                raise TimeoutError("passenger calculation deadline or cancellation")
        bounded=deadline is not None or cancel_event is not None
        if bounded: check_deadline()
        start=integer(start_second,"start second")
        end=self.duration if end_second is None else integer(end_second,"end second",1)
        if not start<end<=self.duration: raise ValueError("invalid passenger window")
        if checkpoint is None:
            if start: raise ValueError("nonzero start needs a checkpoint")
            state=self._initial()
        else:
            try: self._validate_checkpoint(checkpoint,start)
            except (TypeError,KeyError,AttributeError) as exc: raise ValueError("malformed grouped checkpoint") from exc
            state=json.loads(json.dumps(checkpoint,ensure_ascii=False)); state.pop("sha256")
            state["queues"]={key:deque(rows) for key,rows in state["queues"].items()}
        initial={k:state[k] for k in ("entered","platform_arrived","boarded","completed","approaching","queued","onboard_count","waiting_passenger_seconds")}
        initial_age=self._queue_age(state)
        initial_direction_peak=max(state["queue_counts"].values(),default=0)
        initial_loads=copy.deepcopy(state["loads"])
        previous_hist=state["wait_histogram"].copy()
        while state["next_event_index"]<len(self.stream):
            if bounded and state["next_event_index"]%256==0: check_deadline()
            e=self.stream[state["next_event_index"]]; second=e["second"]
            closing=end==self.duration and e["kind"]=="motion" and second==end
            if second>end or second==end and not closing:
                if second==end==self.duration and e["kind"] in {"gate","platform"}:
                    state["next_event_index"]+=1; continue
                break
            state["waiting_passenger_seconds"]+=state["queued"]*(second-state["second"])
            state["second"]=second
            self._process(state,e); state["next_event_index"]+=1
        if bounded: check_deadline()
        state["waiting_passenger_seconds"]+=state["queued"]*(end-state["second"])
        state["second"]=end
        histogram={int(w):n-previous_hist.get(w,0) for w,n in state["wait_histogram"].items() if n-previous_hist.get(w,0)}
        boarded=sum(histogram.values()); full_wait=sum(w*n for w,n in histogram.items())
        rank=(95*boarded+99)//100; cumulative=0; p95=None
        for w,n in sorted(histogram.items()):
            cumulative+=n
            if cumulative>=rank: p95=w; break
        queued=[{**g,"waiting_age_seconds":end-g["platform_second"]} for key in sorted(state["queues"]) for g in state["queues"][key]]
        residual=sum(g["count"]*g["waiting_age_seconds"] for g in queued)
        area=state["waiting_passenger_seconds"]-initial["waiting_passenger_seconds"]
        initial_people=initial["approaching"]+initial["queued"]+initial["onboard_count"]
        new_entries=state["entered"]-initial["entered"]
        completed=state["completed"]-initial["completed"]
        balance=initial_people+new_entries==completed+state["approaching"]+state["queued"]+state["onboard_count"]
        identity=area==full_wait+residual-initial_age
        if not balance or not identity: raise ValueError("grouped passenger conservation failed")
        loads=[row for row in state["load_events"] if start<=row["second"]<=end]
        samples=[row for row in state["queue_samples"] if start<=row["second"]<end]
        queue_events=[row for row in state["queue_events"] if start<=row["second"]<end]
        max_initial=max((n for counts in initial_loads.values() for n in counts),default=0)
        peak=max([max_initial]+[row["load"] for row in loads])
        metrics={"entered":new_entries,"platform_arrived":state["platform_arrived"]-initial["platform_arrived"],
            "boarded":boarded,"completed":completed,"approaching":state["approaching"],"queued":state["queued"],
            "onboard":state["onboard_count"],"initial_passengers":initial_people,
            "initial_approaching":initial["approaching"],"initial_queued":initial["queued"],"initial_onboard":initial["onboard_count"],
            "waiting_passenger_seconds":area,"full_boarded_wait_passenger_seconds":full_wait,
            "mean_full_wait_seconds":number(ratio(Decimal(full_wait),Decimal(boarded))) if boarded else None,
            "p95_full_wait_seconds":p95,"wait_scope":"platform wait; full age for boarding inside this window",
            "residual_queue_age_passenger_seconds":residual,"initial_queue_age_passenger_seconds":initial_age,
            "max_queue_age_seconds":max((g["waiting_age_seconds"] for g in queued),default=0),
            "max_queue":max([initial["queued"]]+[r["queued"] for r in queue_events]),
            "max_platform_direction_queue":max([initial_direction_peak]+[r["max_platform_direction_queue"] for r in queue_events]),"max_car_load":peak,
            "normal_threshold_exceedances":sum(r["threshold_exceeded"] for r in loads),
            "balance":{"initial":initial_people,"entered":new_entries,"completed":completed,
                       "queued":state["queued"],"onboard":state["onboard_count"],"approaching":state["approaching"],"passed":balance},
            "waiting_identity_passed":identity,"cumulative":{"entered":state["entered"],"platform_arrived":state["platform_arrived"],
                "boarded":state["boarded"],"completed":state["completed"],"waiting_passenger_seconds":state["waiting_passenger_seconds"]}}
        approaching=[copy.deepcopy(g) for g in self.by_id.values() if g["turnstile_second"]<end<=g["platform_second"]]
        state["queues"]={key:list(queue) for key,queue in sorted(state["queues"].items())}
        state["sha256"]=PassengerSimulation._seal(state)
        if bounded: check_deadline()
        return {"model_version":MODEL_VERSION,"movement_model_version":self.motion["model_version"],
            "window_start_second":start,"window_end_second":end,"metrics":metrics,"queues":queued,
            "approaching_groups":approaching,"initial_car_loads":initial_loads,"queue_samples":samples,
            "load_events":loads,"checkpoint":state,"release_authorized":False}
