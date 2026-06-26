import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ConfidenceBadge } from "@/components/core/confidence-badge";
import { HeroDemoVideo } from "@/components/video/hero-demo-video";
import { SpeedGauge } from "@/components/charts/charts";
import { tennisProject } from "@/lib/mock/fixtures";
import {
  ArrowUpRight,
  Lock,
  Ruler,
  Sparkles,
  Map,
  Zap,
  CircleDot,
  Users,
  Camera,
} from "lucide-react";

const PILLARS = [
  {
    n: "01",
    icon: Ruler,
    title: "Calibre ton angle, ne le subis pas",
    text: "Homographie 4-coins → mètres réels perspective-corrigés. Vitesse, profondeur et placement fonctionnent même sur les angles rasants où les autres s'effondrent.",
  },
  {
    n: "02",
    icon: Lock,
    title: "Tes données ne quittent jamais ton appareil",
    text: "Pipeline local-first, zéro upload. La confidentialité est une fonctionnalité, pas un coût — pensé pour les clubs et écoles (RGPD).",
  },
  {
    n: "03",
    icon: Sparkles,
    title: "Des chiffres bruts à un plan d'action",
    text: "Narration IA ancrée sur tes données réelles, benchmarks de niveau, diagnostic forces/faiblesses + recommandations. Un coach dans la machine.",
  },
];

const PIPELINE = [
  { icon: Camera, name: "Calibrateur", text: "4-coins semi-auto · 14 keypoints ITF" },
  { icon: Map, name: "Court radar", text: "Minimap top-down métrique exacte" },
  { icon: Zap, name: "FX broadcast", text: "Bloom · HUD · heatmap INFERNO" },
  { icon: CircleDot, name: "Rebonds", text: "F1 0.83 sur cas rasant pathologique" },
  { icon: Users, name: "Tracking P1/P2", text: "Identité persistante, zéro frame perdue" },
];

export default function LandingPage() {
  const p = tennisProject;

  return (
    <div className="overflow-hidden">
      {/* ───────────── HERO ───────────── */}
      <section className="relative">
        {/* animated backdrop — drifting court grid + floating glow orbs */}
        <div aria-hidden className="hero-backdrop">
          <div className="hero-grid" />
          <div
            className="hero-orb"
            style={{ left: "62%", top: "8%", height: "32rem", width: "32rem", background: "var(--color-ball)", opacity: 0.14 }}
          />
          <div
            className="hero-orb"
            style={{ left: "-6%", top: "40%", height: "26rem", width: "26rem", background: "var(--color-clay)", opacity: 0.12, animationDelay: "-8s" }}
          />
        </div>

        {/* oversized ghost word */}
        <div
          aria-hidden
          className="pointer-events-none absolute -right-10 top-24 z-0 select-none font-display text-[28vw] font-extrabold leading-none text-foreground/[0.015] sm:text-[22vw]"
        >
          ACE
        </div>

        <div className="relative z-10 mx-auto grid max-w-6xl grid-cols-1 items-center gap-14 px-4 py-20 sm:px-6 lg:grid-cols-12 lg:py-28">
          <div className="lg:col-span-6">
            <div className="rise mb-6 inline-flex items-center gap-2 rounded-full border border-ball/25 bg-ball/[0.06] px-3 py-1">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-ball opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-ball" />
              </span>
              <span className="eyebrow !text-foreground/80">Analyse · local-first · niveau pro</span>
            </div>

            <h1 className="rise font-display text-[clamp(2.75rem,7vw,5.5rem)] font-extrabold leading-[0.92] tracking-tight" style={{ animationDelay: "60ms" }}>
              Ton match,
              <br />
              disséqué image
              <br />
              par image.
            </h1>

            <p className="rise mt-7 max-w-md text-lg leading-relaxed text-muted-foreground" style={{ animationDelay: "140ms" }}>
              Filmé au téléphone, depuis n&apos;importe quel angle. Vitesse, profondeur, placement,
              momentum — sans cloud, sans crédits, sur n&apos;importe quel appareil.
            </p>

            <div className="rise mt-9 flex flex-wrap items-center gap-3" style={{ animationDelay: "220ms" }}>
              <Button size="lg" render={<Link href="/signup" />} className="group h-11 bg-ball px-5 text-base text-primary-foreground hover:bg-ball/90">
                Analyser ma vidéo
                <ArrowUpRight className="ml-1 h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </Button>
              <Button size="lg" variant="outline" render={<Link href="/login" />} className="h-11 border-foreground/15 px-5 text-base">
                Voir la démo
              </Button>
            </div>

            {/* trust row — what makes CourtSide different (not match stats) */}
            <div className="rise mt-12 flex flex-wrap items-center gap-x-6 gap-y-2 font-mono text-[11px] uppercase tracking-widest text-muted-foreground" style={{ animationDelay: "300ms" }}>
              <span className="flex items-center gap-1.5"><span className="h-1 w-1 rounded-full bg-ball" /> Sans abonnement premium</span>
              <span className="flex items-center gap-1.5"><span className="h-1 w-1 rounded-full bg-ball" /> Tout appareil</span>
              <span className="flex items-center gap-1.5"><span className="h-1 w-1 rounded-full bg-ball" /> 100% privé</span>
            </div>
          </div>

          {/* hero demo video — the real annotated pipeline output (motion + wow) */}
          <div className="rise lg:col-span-6" style={{ animationDelay: "180ms" }}>
            <HeroDemoVideo
              mp4="/demo/hero-tennis.mp4"
              webm="/demo/hero-tennis.webm"
              poster="/demo/hero-tennis-poster.jpg"
              label="analyse en direct"
            />
            <p className="mt-3 text-center font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
              sortie réelle · squelettes verrouillés · radar métrique
            </p>
          </div>
        </div>
      </section>

      {/* ───────────── PILLARS ───────────── */}
      <section className="border-y border-foreground/6 court-grid">
        <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
          <p className="eyebrow mb-3">Pourquoi CourtSide</p>
          <h2 className="mb-12 max-w-2xl font-display text-3xl font-extrabold tracking-tight sm:text-4xl">
            Le seul à occuper le terrain laissé vide.
          </h2>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            {PILLARS.map((pil) => {
              const Icon = pil.icon;
              return (
                <div key={pil.n} className="panel group p-6 transition-transform duration-300 hover:-translate-y-1">
                  <div className="mb-5 flex items-center justify-between">
                    <span className="font-mono text-xs text-muted-foreground">{pil.n}</span>
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-ball/20 bg-ball/10 text-ball transition-colors group-hover:bg-ball/20">
                      <Icon className="h-5 w-5" />
                    </div>
                  </div>
                  <h3 className="mb-2 font-display text-lg font-semibold">{pil.title}</h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">{pil.text}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ───────────── PIPELINE ───────────── */}
      <section id="pipeline" className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
        <div className="mb-10 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="eyebrow mb-3">Sous le capot</p>
            <h2 className="font-display text-3xl font-extrabold tracking-tight sm:text-4xl">Le moteur de vision</h2>
          </div>
          <p className="max-w-xs text-sm text-muted-foreground">Cinq briques de computer-vision qui transforment une vidéo brute en données métriques.</p>
        </div>
        <div className="grid grid-cols-2 gap-px overflow-hidden rounded-2xl border border-foreground/8 bg-foreground/5 lg:grid-cols-5">
          {PIPELINE.map((m, i) => {
            const Icon = m.icon;
            return (
              <div key={m.name} className="group bg-card/80 p-5 transition-colors hover:bg-accent">
                <div className="mb-3 flex items-center justify-between">
                  <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-foreground/8 bg-foreground/5 text-clay transition-colors group-hover:text-ball">
                    <Icon className="h-4 w-4" />
                  </div>
                  <span className="font-mono text-[10px] text-muted-foreground">{String(i + 1).padStart(2, "0")}</span>
                </div>
                <h3 className="font-display text-sm font-semibold">{m.name}</h3>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{m.text}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* ───────────── HONEST STATS ───────────── */}
      <section id="demo" className="border-y border-foreground/6 court-bg">
        <div className="mx-auto grid max-w-6xl grid-cols-1 items-center gap-12 px-4 py-20 sm:px-6 lg:grid-cols-2">
          <div>
            <p className="eyebrow mb-3">Honnêteté calibrée</p>
            <h2 className="font-display text-3xl font-extrabold tracking-tight sm:text-4xl">
              Des stats honnêtes,
              <br />pas des promesses.
            </h2>
            <p className="mt-4 max-w-md text-muted-foreground">
              Chaque métrique porte un badge de confiance. Le produit ne ment jamais sur ce qu&apos;il
              mesure.
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              <ConfidenceBadge source="homography(high)" showTooltip={false} />
              <ConfidenceBadge source="scalar" showTooltip={false} />
              <ConfidenceBadge tier="approx" showTooltip={false} />
            </div>
            <Button className="mt-8 bg-ball text-primary-foreground hover:bg-ball/90" render={<Link href="/signup" />}>
              Essayer gratuitement
            </Button>
          </div>
          <div className="panel-raised p-8">
            <SpeedGauge summary={p.stats.summary} />
          </div>
        </div>
      </section>

      {/* ───────────── CTA ───────────── */}
      <section className="relative mx-auto max-w-6xl px-4 py-24 text-center sm:px-6">
        <p className="eyebrow mb-4">1 vidéo / mois offerte</p>
        <h2 className="mx-auto max-w-2xl font-display text-4xl font-extrabold leading-tight tracking-tight sm:text-5xl">
          Prêt à voir ton jeu comme un pro ?
        </h2>
        <p className="mx-auto mt-3 max-w-md text-muted-foreground">
          Toutes les stats, dès la première analyse. Pas de carte bancaire, pas de crédits.
        </p>
        <Button size="lg" className="group mt-8 h-12 bg-ball px-6 text-base text-primary-foreground hover:bg-ball/90" render={<Link href="/signup" />}>
          Commencer maintenant
          <ArrowUpRight className="ml-1 h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
        </Button>
      </section>
    </div>
  );
}
