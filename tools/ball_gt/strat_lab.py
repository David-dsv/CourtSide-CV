"""Strategy lab: replay BOTH caches (demo3 ball-GT + felix bounce-GT) under a
candidate selection strategy and report demo3 coverage/CORRECT AND felix bounce
F1 in one shot. The two-clip thermometer that catches demo3-overfit instantly
(felix is the parasite-heavy oblique adversary; demo3 the clean broadcast).

  python tools/ball_gt/strat_lab.py --strat NAME --conf C --imgsz 1280|1920

This is iteration-only tooling (not prod). Prod faithfulness is re-checked LIVE.
"""
import argparse, json, importlib.util, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
ROOT = Path(__file__).resolve().parents[2]

# felix bounce harness (prod-faithful tracker + matcher)
_spec = importlib.util.spec_from_file_location("_tbr", str(ROOT/"tests"/"test_bounce_regression.py"))
_tbr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_tbr)
from vision.bounce import detect_bounces_from_trajectory, smooth_ball_trajectory

DEMO3_GT = json.load(open(ROOT/"tests"/"fixtures"/"ball_gt"/"tennis_demo3.ball_gt.json"))
DEMO3_GTBF = {f["frame"]: f["ball"] for f in DEMO3_GT["frames"]
              if f.get("verified") and f.get("visible",True) and f.get("ball")}
FELIX_BGT = [b["frame"] for b in json.load(open(ROOT/"tests"/"fixtures"/"bounces"/"felix.bounces.json"))["bounces"]]


def demo3_score(track, fw, fps):
    tol=0.02*fw; start_n=int(4.0*fps)
    cov=corr=n=cov_s=corr_s=n_s=0
    for fidx,g in DEMO3_GTBF.items():
        if fidx>=len(track): continue
        n+=1; iss=fidx<start_n; n_s+=iss
        t=track[fidx]
        if t is not None:
            cov+=1; cov_s+=iss
            if np.hypot(t[0]-g[0],t[1]-g[1])<tol: corr+=1; corr_s+=iss
    return dict(cov=cov,corr=corr,n=n,cov_s=cov_s,corr_s=corr_s,n_s=n_s,
               cov_p=100*cov/max(1,n),corr_p=100*corr/max(1,n),
               cov_sp=100*cov_s/max(1,n_s),corr_sp=100*corr_s/max(1,n_s))


def felix_bounce_f1(track, fw, fh, fps):
    smoothed = smooth_ball_trajectory(track, max_gap=int(fps*0.4))
    is_real=[c is not None for c in track]
    bounces = detect_bounces_from_trajectory(smoothed,[0.0]*len(smoothed),fps,fh,fw,raw_centers=track,is_real=is_real)
    pred=[b[0] for b in bounces]; tol=round(0.15*fps)
    tp,fp,fn=_tbr._match(pred, FELIX_BGT, tol)
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return (2*p*r/(p+r) if p+r else 0), tp, fp, fn


def load_cache(clip, imgsz):
    return json.loads((ROOT/"tests"/"fixtures"/"ball_gt"/f"{clip}_det_cache_{imgsz}.json").read_text())


def prod_track(frames, conf, fw, fh, fps, **kw):
    """The PROD-FAITHFUL tracker (= test_bounce_regression._track): conf-filter
    then int-rounded Kalman. The AUTHORITATIVE felix floor gate — exactly what the
    shipped BallKalmanFilter does. Any felix decision is made against THIS."""
    raw = [[(int(cx),int(cy),s) for cx,cy,s in fr if s>=conf] for fr in frames]
    return _tbr._track(raw, fw, fh, fps)


def run_strat(strat_fn, conf, imgsz, felix_fn=None, **kw):
    """felix_fn: the tracker used for the felix bounce gate. Defaults to the
    PROD-FAITHFUL prod_track (int-rounded), what test_bounce_regression asserts.
    Pass the SAME strat_fn for felix only when measuring that novel strategy's
    felix behaviour explicitly."""
    dc = load_cache("demo3", imgsz)
    df = [[(c["cx"],c["cy"],c["score"]) for c in fr] for fr in dc["frames"]]
    dtrack = strat_fn(df, conf, dc["width"], dc["height"], dc["fps"], **kw)
    ds = demo3_score(dtrack, dc["width"], dc["fps"])

    fc = load_cache("felix", imgsz)
    fn_track = felix_fn if felix_fn is not None else prod_track
    ff = [[(c["cx"],c["cy"],c["score"]) for c in fr] for fr in fc["frames"]]
    ftrack = fn_track(ff, conf, fc["width"], fc["height"], fc["fps"], **kw)
    ftrack = [(int(c[0]),int(c[1])) if c is not None else None for c in ftrack]
    f1,tp,fp,fn = felix_bounce_f1(ftrack, fc["width"], fc["height"], fc["fps"])
    return ds, (f1,tp,fp,fn)


def report(name, ds, fx):
    f1,tp,fp,fn = fx
    flag = "OK" if f1>=0.72 else "FELIX-FAIL"
    print(f"[{name}] demo3 cov {ds['cov_p']:.1f}% CORR {ds['corr_p']:.1f}% | "
          f"START cov {ds['cov_sp']:.1f}% CORR {ds['corr_sp']:.1f}% || "
          f"felix bounce F1={f1:.3f} (TP={tp} FP={fp} FN={fn}) [{flag}]", flush=True)
