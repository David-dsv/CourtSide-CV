"""
Visual pose-GT annotator — a tiny zero-dependency local web app to correct the
FAR player's bounding box (the SOTA blocker: pipeline locks the far player only
~23% of frames). Click-drag a box on each frame, arrow-key navigate, auto-save.

Pipeline
--------
1. (once) generate the prefilled template + extract frames:
     python tools/pose_gt/make_pose_gt_template.py --step 5
     python tools/pose_gt/annotate_server.py --extract-frames
2. annotate:
     python tools/pose_gt/annotate_server.py
   → open http://localhost:8011  (auto-opens). Draw the FAR box, press N/P or
     arrows to move, S to save (also auto-saves on every edit). Set 'present'
     off if the far player is truly out of frame. 'verified' is set true the
     moment you touch a frame.

It edits tests/fixtures/pose_gt/tennis_demo3.pose_gt.json IN PLACE (writes a
.bak first). Only stdlib — no Flask, no deps.
"""
import argparse
import json
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"
FRAMES_DIR = ROOT / "tests" / "fixtures" / "pose_gt" / "frames_demo3"
PORT = 8011

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Pose GT — far box</title>
<style>
 body{margin:0;background:#0b0a08;color:#f4efe6;font:14px/1.4 system-ui}
 #bar{position:fixed;top:0;left:0;right:0;padding:8px 12px;background:#17150f;
   display:flex;gap:14px;align-items:center;z-index:10;box-shadow:0 2px 8px #0008}
 #wrap{margin-top:46px;display:flex;justify-content:center}
 canvas{cursor:crosshair;max-width:100vw;border:1px solid #333}
 button{background:#d8f64a;color:#16150f;border:0;border-radius:6px;padding:6px 12px;
   font-weight:700;cursor:pointer}
 button.sec{background:#2e2a22;color:#f4efe6}
 .pill{padding:3px 9px;border-radius:99px;font-weight:700}
 .ok{background:#2f7d4f}.no{background:#e2603a}.todo{background:#6b6457}
 kbd{background:#2e2a22;border-radius:4px;padding:1px 6px;font-family:monospace}
 #counter{font-variant-numeric:tabular-nums}
</style></head><body>
<div id=bar>
 <button class=sec onclick=go(-1)>◀ P</button>
 <span id=counter></span>
 <button class=sec onclick=go(1)>N ▶</button>
 <span id=status class="pill todo">…</span>
 <label><input type=checkbox id=present checked onchange=togglePresent()> far présent</label>
 <button onclick=validateAsIs()>✓ Point OK (V)</button>
 <button onclick=clearPt()>Pas de joueur far (X)</button>
 <button onclick=save()>Sauver (S)</button>
 <span style=opacity:.6>CLIC = centre du joueur far · <kbd>V</kbd> valider+suivant · <kbd>←→</kbd> nav · <kbd>X</kbd> absent · <kbd>S</kbd> save</span>
 <span id=cov style=margin-left:auto;opacity:.8></span>
</div>
<div id=wrap><canvas id=c></canvas></div>
<script>
let data=null, frames=[], i=0, img=new Image(), pt=null, present=true, scale=1;
const c=document.getElementById('c'), x=c.getContext('2d');
function boxCenter(b){ return b?[Math.round((b[0]+b[2])/2*10)/10, Math.round((b[1]+b[3])/2*10)/10]:null; }
async function load(){ data=await (await fetch('/gt')).json(); frames=data.frames; show(); }
function cur(){ return frames[i]; }
function show(){
  const f=cur();
  present = f.p1_far.present!==false;
  // point = explicit verified center if set, else seed from the prefilled box center
  pt = f.p1_far.center ? f.p1_far.center.slice() : boxCenter(f.p1_far.box);
  document.getElementById('present').checked=present;
  img.onload=()=>{ const W=Math.min(1280, img.width); scale=W/img.width;
    c.width=img.width*scale; c.height=img.height*scale; draw(); };
  img.src='/frame/'+f.frame+'?'+Date.now();
  document.getElementById('counter').textContent=`frame ${f.frame}  (${i+1}/${frames.length})`;
  updStatus(); updCov();
}
function draw(){
  x.drawImage(img,0,0,c.width,c.height);
  // near box (ref, dim) so you don't confuse the two players
  const nb=cur().p2_near.box;
  if(nb){ x.strokeStyle='#5fb0a8';x.lineWidth=1.5;x.strokeRect(nb[0]*scale,nb[1]*scale,(nb[2]-nb[0])*scale,(nb[3]-nb[1])*scale);
    x.fillStyle='#5fb0a8';x.fillText('near (P2)',nb[0]*scale+3,nb[1]*scale-4); }
  if(pt && present){ const X=pt[0]*scale, Y=pt[1]*scale;
    x.strokeStyle='#d8f64a';x.lineWidth=2.5;
    x.beginPath();x.arc(X,Y,9,0,7);x.stroke();
    x.beginPath();x.moveTo(X-15,Y);x.lineTo(X+15,Y);x.moveTo(X,Y-15);x.lineTo(X,Y+15);x.stroke();
    x.fillStyle='#d8f64a';x.fillText('FAR (P1)',X+13,Y-11); }
}
function updStatus(){ const f=cur(); const s=document.getElementById('status');
  if(!present){s.className='pill no';s.textContent='far ABSENT';}
  else if(f.p1_far.verified){s.className='pill ok';s.textContent='vérifié ✓';}
  else {s.className='pill todo';s.textContent='à vérifier';} }
function updCov(){ const v=frames.filter(f=>f.p1_far.verified).length;
  const p=frames.filter(f=>f.p1_far.verified&&f.p1_far.present!==false&&f.p1_far.center).length;
  document.getElementById('cov').textContent=`vérifié ${v}/${frames.length} · far présent+point ${p}`; }
c.onclick=e=>{ if(!present){present=true; document.getElementById('present').checked=true;}
  const r=c.getBoundingClientRect();
  pt=[Math.round((e.clientX-r.left)/scale*10)/10, Math.round((e.clientY-r.top)/scale*10)/10];
  commit(); draw(); };
function commit(){ const f=cur(); f.p1_far.center=present?pt:null; f.p1_far.present=present;
  f.p1_far.verified=true; updStatus(); updCov(); save(); }
function validateAsIs(){ const f=cur(); f.p1_far.center=present?pt:null; f.p1_far.present=present;
  f.p1_far.verified=true; updStatus(); updCov(); save(); go(1); }
function clearPt(){ present=false; pt=null; document.getElementById('present').checked=false;
  commit(); draw(); }
function togglePresent(){ present=document.getElementById('present').checked;
  if(!present) pt=null; commit(); draw(); }
function go(d){ i=Math.max(0,Math.min(frames.length-1,i+d)); show(); }
async function save(){ await fetch('/save',{method:'POST',headers:{'content-type':'application/json'},
  body:JSON.stringify(data)}); }
document.onkeydown=e=>{ if(e.key==='ArrowRight'||e.key==='n')go(1);
  else if(e.key==='ArrowLeft'||e.key==='p')go(-1);
  else if(e.key==='s'){e.preventDefault();save();}
  else if(e.key==='v')validateAsIs();
  else if(e.key==='x')clearPt(); };
load();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif self.path == "/gt":
            self._send(200, "application/json", GT.read_bytes())
        elif self.path.startswith("/frame/"):
            fid = int(self.path.split("/")[2].split("?")[0])
            p = FRAMES_DIR / f"{fid:05d}.jpg"
            if p.exists():
                self._send(200, "image/jpeg", p.read_bytes())
            else:
                self._send(404, "text/plain", b"frame missing - run --extract-frames")
        else:
            self._send(404, "text/plain", b"nope")

    def do_POST(self):
        if self.path == "/save":
            n = int(self.headers["Content-Length"])
            body = self.rfile.read(n)
            if GT.exists():
                GT.with_suffix(".json.bak").write_bytes(GT.read_bytes())
            GT.write_bytes(body)
            self._send(200, "application/json", b'{"ok":true}')
        else:
            self._send(404, "text/plain", b"nope")


def extract_frames():
    """Decode the needed frames one-by-one (robust: a 200+ term select filter
    overflows ffmpeg's parser). We seek to each frame and grab 1 jpg — a bit more
    ffmpeg launches but rock-solid, and only the SAMPLED frames are touched."""
    if not GT.exists():
        print(f"ERROR: {GT} missing — run make_pose_gt_template.py first.")
        sys.exit(1)
    doc = json.load(open(GT))
    fps = float(doc.get("fps", 50.0))
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    frames = [f["frame"] for f in doc["frames"]]
    todo = [n for n in frames if not (FRAMES_DIR / f"{n:05d}.jpg").exists()]
    if not todo:
        print(f"All {len(frames)} frames already extracted.")
        return
    print(f"Extracting {len(todo)} frames from {CLIP} → {FRAMES_DIR} ...")
    for k, n in enumerate(todo):
        ts = n / fps
        # -ss before -i = fast keyframe seek + accurate decode to the exact frame
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{ts:.4f}", "-i", str(CLIP),
            "-frames:v", "1", "-q:v", "3",
            str(FRAMES_DIR / f"{n:05d}.jpg")
        ], check=True)
        if (k + 1) % 25 == 0 or k + 1 == len(todo):
            print(f"  {k+1}/{len(todo)}")
    print(f"Done: {len(todo)} frames extracted.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract-frames", action="store_true",
                    help="extract the to-annotate frames as jpg (run once before annotating)")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    if args.extract_frames:
        extract_frames()
        return
    if not GT.exists():
        print(f"ERROR: {GT} missing — run make_pose_gt_template.py first.")
        sys.exit(1)
    if not any(FRAMES_DIR.glob("*.jpg")):
        print("No frames extracted yet — running --extract-frames first ...")
        extract_frames()
    srv = HTTPServer(("127.0.0.1", args.port), H)
    url = f"http://localhost:{args.port}"
    print(f"Pose-GT annotator → {url}  (Ctrl-C to stop). Edits save to {GT.name} (.bak kept).")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
