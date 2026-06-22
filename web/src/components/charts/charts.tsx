"use client";

import { depthHex, qualityHex } from "@/lib/format";
import type { Summary } from "@/lib/types";

/** Broadcast-style speed gauge: arc green→amber→red, needle on avg. */
export function SpeedGauge({ summary }: { summary: Summary }) {
  const max = 200; // km/h scale
  const v = Math.min(summary.speed_kmh.avg, max);
  const pct = v / max;
  const startAngle = 150; // deg
  const sweep = 240; // deg
  const angle = startAngle + sweep * pct;

  const cx = 100, cy = 95, r = 78;
  const rad = (a: number) => (a * Math.PI) / 180;
  const pt = (a: number, rad2: number) => [
    cx + rad2 * Math.cos(rad(a)),
    cy + rad2 * Math.sin(rad(a)),
  ];

  const arc = (a0: number, a1: number, color: string) => {
    const [x0, y0] = pt(a0, r);
    const [x1, y1] = pt(a1, r);
    const large = a1 - a0 > 180 ? 1 : 0;
    return `<path d="M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1}" stroke="${color}" stroke-width="14" fill="none" stroke-linecap="round" />`;
  };

  const a0 = startAngle;
  const aGreen = startAngle + sweep * 0.4;
  const aAmber = startAngle + sweep * 0.7;
  const a1 = startAngle + sweep;
  const [nx, ny] = pt(angle, r - 14);

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 130" className="w-full max-w-[220px]">
        <g dangerouslySetInnerHTML={{
          __html:
            arc(a0, aGreen, "#d8f64a") +
            arc(aGreen, aAmber, "#f2b441") +
            arc(aAmber, a1, "#e2603a"),
        }} />
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="#f5faff" strokeWidth={3} strokeLinecap="round" />
        <circle cx={cx} cy={cy} r={5} fill="#f5faff" />
      </svg>
      <div className="-mt-6 text-center">
        <div className="stat-numeral text-5xl text-ball">
          {summary.speed_kmh.avg.toFixed(0)}
        </div>
        <div className="mt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          km/h moyen
        </div>
        <div className="mt-2 font-mono text-[11px] text-muted-foreground">
          max {summary.speed_kmh.max.toFixed(0)} · min {summary.speed_kmh.min.toFixed(0)}
        </div>
      </div>
    </div>
  );
}

/** Donut showing deep/mid/short bounce distribution. */
export function DepthDonut({ summary }: { summary: Summary }) {
  const { deep, mid, short } = summary.depth;
  const total = deep + mid + short || 1;
  const segments = [
    { v: deep, color: depthHex("deep"), label: "Profond" },
    { v: mid, color: depthHex("mid"), label: "Moyen" },
    { v: short, color: depthHex("short"), label: "Court" },
  ];
  const cx = 60, cy = 60, r = 50, stroke = 16;
  const circ = 2 * Math.PI * r;
  let offset = 0;
  return (
    <div className="flex items-center gap-4">
      <svg viewBox="0 0 120 120" className="h-32 w-32">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={stroke} />
        {segments.map((s) => {
          const len = (s.v / total) * circ;
          const el = (
            <circle
              key={s.label}
              cx={cx}
              cy={cy}
              r={r}
              fill="none"
              stroke={s.color}
              strokeWidth={stroke}
              strokeDasharray={`${len} ${circ - len}`}
              strokeDashoffset={-offset}
              transform={`rotate(-90 ${cx} ${cy})`}
              strokeLinecap="butt"
            />
          );
          offset += len;
          return el;
        })}
        <text x={cx} y={cy - 2} textAnchor="middle" className="fill-foreground font-semibold" fontSize={22}>
          {total}
        </text>
        <text x={cx} y={cy + 14} textAnchor="middle" className="fill-muted-foreground" fontSize={9}>
          rebonds
        </text>
      </svg>
      <div className="flex flex-col gap-1.5 text-sm">
        {segments.map((s) => (
          <div key={s.label} className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: s.color }} />
            <span className="text-muted-foreground">{s.label}</span>
            <span className="ml-auto font-medium tabular-nums">{s.v}</span>
            <span className="w-10 text-right text-xs text-muted-foreground">
              {Math.round((s.v / total) * 100)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Forehand vs backhand bar. */
export function StrokesBar({ summary }: { summary: Summary }) {
  const { forehand, backhand } = summary.strokes;
  const total = forehand + backhand || 1;
  const fhPct = (forehand / total) * 100;
  return (
    <div>
      <div className="flex h-8 w-full overflow-hidden rounded-lg">
        <div
          className="flex items-center justify-center text-xs font-medium text-background"
          style={{ width: `${fhPct}%`, background: "#d8f64a" }}
        >
          {forehand > 0 && `${forehand}`}
        </div>
        <div
          className="flex items-center justify-center text-xs font-medium text-background"
          style={{ width: `${100 - fhPct}%`, background: "#5fb0a8" }}
        >
          {backhand > 0 && `${backhand}`}
        </div>
      </div>
      <div className="mt-1.5 flex justify-between text-xs text-muted-foreground">
        <span>Coup droit ({Math.round(fhPct)}%)</span>
        <span>Revers ({Math.round(100 - fhPct)}%)</span>
      </div>
    </div>
  );
}

/** Quality sparkline per bounce (chronological). */
export function QualitySparkline({ bounces }: { bounces: import("@/lib/types").Bounce[] }) {
  const qs = bounces.map((b) => b.quality);
  if (qs.length === 0) return <div className="text-sm text-muted-foreground">Aucune donnée</div>;
  const w = 280, h = 70, pad = 6;
  const max = 100;
  const stepX = (w - pad * 2) / Math.max(qs.length - 1, 1);
  const pts = qs.map((q, i) => [pad + i * stepX, h - pad - (q / max) * (h - pad * 2)]);
  const line = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
  const avg = qs.reduce((a, b) => a + b, 0) / qs.length;
  const avgY = h - pad - (avg / max) * (h - pad * 2);
  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
        <line x1={pad} y1={avgY} x2={w - pad} y2={avgY} stroke="#d8f64a" strokeWidth={1} strokeDasharray="3 3" opacity={0.5} />
        <path d={line} fill="none" stroke="#d8f64a" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {pts.map((p, i) => (
          <circle key={i} cx={p[0]} cy={p[1]} r={2.5} fill={qualityHex(qs[i])} />
        ))}
      </svg>
      <div className="mt-1 flex justify-between text-xs text-muted-foreground">
        <span>1er rebond</span>
        <span>moy. {avg.toFixed(0)}/100</span>
        <span>dernier</span>
      </div>
    </div>
  );
}
