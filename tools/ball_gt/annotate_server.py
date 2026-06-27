"""
Visual BALL-GT annotator — a tiny zero-dependency local web app to correct the
ball position per frame (the SOTA blocker is now ball-tracking quality, esp.
cold-start at the clip start). Click the ball, arrow-key navigate, auto-save.

Pipeline
--------
1. (once) generate the prefilled template + extract frames:
     python tools/ball_gt/make_ball_gt_template.py
     python tools/ball_gt/annotate_server.py --extract-frames
2. annotate:
     python tools/ball_gt/annotate_server.py
   → open http://localhost:8012 (auto-opens). CLICK the ball. V = current point
     OK + next. X = ball NOT visible (occluded/off-frame). ←/→ navigate. The
     previous + next frames' ball points are shown faint (the ball moves fast —
     they hint the path). Auto-saves on every edit.

Edits tests/fixtures/ball_gt/tennis_demo3.ball_gt.json in place (.bak kept).
Only stdlib — no deps.
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
GT = ROOT / "tests" / "fixtures" / "ball_gt" / "tennis_demo3.ball_gt.json"
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"
FRAMES_DIR = ROOT / "tests" / "fixtures" / "ball_gt" / "frames_demo3"
PORT = 8012

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Ball GT</title>
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
 <button onclick=validateAsIs()>✓ Point OK (V)</button>
 <button onclick=clearPt()>Balle invisible (X)</button>
 <span style=opacity:.6>CLIC = position de la BALLE · <kbd>V</kbd> valider+suivant · <kbd>←→</kbd> nav · <kbd>X</kbd> invisible</span>
 <span id=cov style=margin-left:auto;opacity:.8></span>
</div>
<div id=wrap><canvas id=c></canvas></div>
<script>
let data=null, frames=[], i=0, img=new Image(), pt=null, visible=true, scale=1;
const c=document.getElementById('c'), x=c.getContext('2d');
async function load(){ data=await (await fetch('/gt')).json(); frames=data.frames; show(); }
function cur(){ return frames[i]; }
function show(){
  const f=cur();
  visible = f.visible!==false;
  pt = f.ball ? f.ball.slice() : null;
  img.onload=()=>{ const W=Math.min(1280, img.width); scale=W/img.width;
    c.width=img.width*scale; c.height=img.height*scale; draw(); };
  img.src='/frame/'+f.frame+'?'+Date.now();
  document.getElementById('counter').textContent=`frame ${f.frame}  (${i+1}/${frames.length})`;
  updStatus(); updCov();
}
function ghost(idx, col){ // draw prev/next ball faint to hint the fast-moving path
  if(idx<0||idx>=frames.length) return;
  const b=frames[idx].ball; if(!b||frames[idx].visible===false) return;
  x.strokeStyle=col; x.lineWidth=1; x.beginPath();
  x.arc(b[0]*scale,b[1]*scale,6,0,7); x.stroke();
}
function draw(){
  x.drawImage(img,0,0,c.width,c.height);
  ghost(i-1,'#5fb0a8aa'); ghost(i+1,'#e2603aaa');  // prev teal, next clay (faint)
  if(pt && visible){ const X=pt[0]*scale, Y=pt[1]*scale;
    x.strokeStyle='#d8f64a';x.lineWidth=2.5;
    x.beginPath();x.arc(X,Y,8,0,7);x.stroke();
    x.beginPath();x.moveTo(X-13,Y);x.lineTo(X+13,Y);x.moveTo(X,Y-13);x.lineTo(X,Y+13);x.stroke();
    x.fillStyle='#d8f64a';x.fillText('BALL',X+11,Y-9); }
}
function updStatus(){ const f=cur(); const s=document.getElementById('status');
  if(!visible){s.className='pill no';s.textContent='balle INVISIBLE';}
  else if(f.verified){s.className='pill ok';s.textContent='vérifié ✓';}
  else {s.className='pill todo';s.textContent='à vérifier';} }
function updCov(){ const v=frames.filter(f=>f.verified).length;
  const p=frames.filter(f=>f.verified&&f.visible!==false&&f.ball).length;
  document.getElementById('cov').textContent=`vérifié ${v}/${frames.length} · balle visible+point ${p}`; }
c.onclick=e=>{ const r=c.getBoundingClientRect();
  visible=true;
  pt=[Math.round((e.clientX-r.left)/scale*10)/10, Math.round((e.clientY-r.top)/scale*10)/10];
  commit(); draw(); };
function commit(){ const f=cur(); f.ball=visible?pt:null; f.visible=visible;
  f.verified=true; updStatus(); updCov(); save(); }
function validateAsIs(){ const f=cur(); f.ball=visible?pt:null; f.visible=visible;
  f.verified=true; updStatus(); updCov(); save(); go(1); }
function clearPt(){ visible=false; pt=null; commit(); draw(); }
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
    if not GT.exists():
        print(f"ERROR: {GT} missing — run make_ball_gt_template.py first.")
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
    ap.add_argument("--extract-frames", action="store_true")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    if args.extract_frames:
        extract_frames()
        return
    if not GT.exists():
        print(f"ERROR: {GT} missing — run make_ball_gt_template.py first.")
        sys.exit(1)
    if not any(FRAMES_DIR.glob("*.jpg")):
        print("No frames extracted yet — running --extract-frames first ...")
        extract_frames()
    srv = HTTPServer(("127.0.0.1", args.port), H)
    url = f"http://localhost:{args.port}"
    print(f"Ball-GT annotator → {url}  (Ctrl-C to stop). Edits save to {GT.name} (.bak kept).")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
