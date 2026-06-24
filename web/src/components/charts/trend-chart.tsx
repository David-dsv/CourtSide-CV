import Link from "next/link";
import type { TrendPoint } from "@/lib/mock/player-profile";

/**
 * Multi-series trend chart across matches: speed (km/h) and quality (/100),
 * each normalized to its own 0..1 scale and drawn as its own line so the two
 * trends can be read together. Each data point is a clickable marker that
 * jumps to that match's overview page (the multi-session view is global; a
 * specific moment lives inside a specific project).
 *
 * Pure SVG (server-renderable), in the same style as QualitySparkline.
 */
export function TrendChart({
  matches,
  projectId,
}: {
  matches: TrendPoint[];
  /** current project id (highlighted marker) */
  projectId: string;
}) {
  if (matches.length === 0) {
    return <div className="text-sm text-muted-foreground">Aucun match à comparer.</div>;
  }

  const w = 520,
    h = 160,
    padX = 28,
    padY = 18;
  const innerW = w - padX * 2;
  const innerH = h - padY * 2;

  const speeds = matches.map((m) => m.speedKmh);
  const quals = matches.map((m) => m.quality);
  const sMin = Math.min(...speeds),
    sMax = Math.max(...speeds);
  const qMin = Math.min(...quals),
    qMax = Math.max(...quals);
  // avoid divide-by-zero when all values equal
  const sRange = sMax - sMin || 1;
  const qRange = qMax - qMin || 1;

  const stepX = matches.length === 1 ? 0 : innerW / (matches.length - 1);
  const x = (i: number) => padX + i * stepX;
  const ySpeed = (v: number) => padY + innerH - ((v - sMin) / sRange) * innerH;
  const yQual = (v: number) => padY + innerH - ((v - qMin) / qRange) * innerH;

  const line = (ys: number[]) =>
    ys.map((yy, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${yy.toFixed(1)}`).join(" ");

  const speedPath = line(matches.map((m) => ySpeed(m.speedKmh)));
  const qualPath = line(matches.map((m) => yQual(m.quality)));

  const fmtDate = (iso: string) => {
    const d = new Date(iso);
    return `${d.getDate().toString().padStart(2, "0")}/${(d.getMonth() + 1).toString().padStart(2, "0")}`;
  };

  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
        {/* baseline grid */}
        <line x1={padX} y1={padY + innerH} x2={w - padX} y2={padY + innerH} stroke="rgba(255,255,255,0.08)" strokeWidth={1} />

        {/* speed line (ball yellow) */}
        <path d={speedPath} fill="none" stroke="#d8f64a" strokeWidth={2.5} strokeLinejoin="round" strokeLinecap="round" />
        {/* quality line (cyan) */}
        <path d={qualPath} fill="none" stroke="#5fb0a8" strokeWidth={2.5} strokeLinejoin="round" strokeLinecap="round" strokeDasharray="1 0" />

        {/* markers — clickable to the match */}
        {matches.map((m, i) => {
          const isCurrent = m.projectId === projectId;
          return (
            <g key={m.projectId}>
              <Link href={`/projects/${m.projectId}/overview`}>
                <title>{`${fmtDate(m.date)} · ${Math.round(m.speedKmh)} km/h · Q${Math.round(m.quality)}`}</title>
                <circle
                  cx={x(i)}
                  cy={ySpeed(m.speedKmh)}
                  r={isCurrent ? 5 : 3.5}
                  fill="#d8f64a"
                  stroke={isCurrent ? "#fff" : "none"}
                  strokeWidth={isCurrent ? 1.5 : 0}
                  className="cursor-pointer transition-transform hover:scale-125"
                />
              </Link>
              <Link href={`/projects/${m.projectId}/overview`}>
                <circle
                  cx={x(i)}
                  cy={yQual(m.quality)}
                  r={isCurrent ? 5 : 3.5}
                  fill="#5fb0a8"
                  stroke={isCurrent ? "#fff" : "none"}
                  strokeWidth={isCurrent ? 1.5 : 0}
                  className="cursor-pointer transition-transform hover:scale-125"
                />
              </Link>
              <text x={x(i)} y={h - 4} textAnchor="middle" fontSize={9} className="fill-muted-foreground" fontFamily="monospace">
                {fmtDate(m.date)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-4 rounded-full bg-ball" /> Vitesse (km/h)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-4 rounded-full bg-court-cyan" /> Qualité (/100)
        </span>
        <span className="ml-auto">Cliquez un point pour ouvrir le match.</span>
      </div>
    </div>
  );
}
