import Link from "next/link";
import { Button } from "@/components/ui/button";
import { GlassCard } from "@/components/core/glass-card";
import { Check } from "lucide-react";
import type { Metadata } from "next";

export const metadata: Metadata = { title: "Tarifs — CourtSide" };

const PLANS = [
  {
    name: "Free",
    price: "0 €",
    period: "/mois",
    tagline: "Pour découvrir",
    features: ["1 vidéo / mois", "Toutes les stats amateur", "Court radar + heatmap", "Narration de base", "Local-first"],
    cta: "Commencer",
    href: "/signup",
    highlight: false,
  },
  {
    name: "Starter",
    price: "19 €",
    period: "/mois",
    tagline: "Pour le joueur régulier",
    features: ["5 vidéos / mois", "Calibration oblique 4-coins", "Benchmarks UTR", "Export PDF", "Mode local-first (Tauri)"],
    cta: "Choisir Starter",
    href: "/signup",
    highlight: true,
  },
  {
    name: "Coach",
    price: "49 €",
    period: "/mois",
    tagline: "Pour coachs & academies",
    features: ["50 vidéos / mois", "Multi-élèves + comparaison", "Plan coach complet", "Calibration avancée", "Support prioritaire"],
    cta: "Choisir Coach",
    href: "/signup",
    highlight: false,
  },
];

const FAQ = [
  { q: "Vidéo illimitée ?", a: "Pas de crédits ni de compteur par frappe. Chaque palier inclut un quota de vidéos/mois, l'analyse de chaque vidéo est illimitée." },
  { q: "Mes vidéos partent-elles sur vos serveurs ?", a: "Non. Le pipeline est local-first par défaut. Le mode cloud n'est activé qu'aux paliers Academy+ et reste optionnel." },
  { q: "Ça marche sur Android ?", a: "Oui — le pipeline Python tourne sur CPU/MPS/CUDA. Le web fonctionne sur tout navigateur, l'app desktop (Tauri) sur macOS/Windows/Linux." },
  { q: "Et si l'angle de ma vidéo est mauvais ?", a: "CourtSide calibre votre angle tel quel (homographie 4-coins). Pas besoin de repositionner votre téléphone." },
];

export default function PricingPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
      <div className="mb-14 text-center">
        <p className="eyebrow mb-3">Tarifs</p>
        <h1 className="font-display text-4xl font-extrabold tracking-tight sm:text-5xl">
          Simples. Sans crédits.
        </h1>
        <p className="mx-auto mt-3 max-w-md text-muted-foreground">
          Flat-rate. Analysez chaque vidéo en profondeur, sans anxiété de compteur.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        {PLANS.map((plan) => (
          <GlassCard
            key={plan.name}
            strong={plan.highlight}
            glow={plan.highlight ? "green" : "none"}
            className={plan.highlight ? "border-ball/40" : ""}
          >
            {plan.highlight && (
              <div className="mb-3 inline-flex rounded-full bg-ball px-3 py-1 font-mono text-[10px] font-semibold uppercase tracking-widest text-primary-foreground">
                Le plus populaire
              </div>
            )}
            <h3 className="font-display text-xl font-extrabold">{plan.name}</h3>
            <p className="text-sm text-muted-foreground">{plan.tagline}</p>
            <div className="mt-4 flex items-baseline gap-1">
              <span className="stat-numeral text-5xl">{plan.price}</span>
              <span className="font-mono text-xs text-muted-foreground">{plan.period}</span>
            </div>
            <Button
              className={`mt-6 w-full ${plan.highlight ? "bg-ball text-primary-foreground hover:bg-ball/90" : ""}`}
              variant={plan.highlight ? "default" : "outline"}
              render={<Link href={plan.href} />}
            >
              {plan.cta}
            </Button>
            <ul className="mt-6 space-y-2.5 text-sm">
              {plan.features.map((f) => (
                <li key={f} className="flex items-start gap-2">
                  <Check className="mt-0.5 h-4 w-4 shrink-0 text-ball" />
                  <span>{f}</span>
                </li>
              ))}
            </ul>
          </GlassCard>
        ))}
      </div>

      <div className="mt-16">
        <h2 className="mb-6 text-center text-2xl font-semibold">Questions fréquentes</h2>
        <div className="mx-auto max-w-2xl space-y-4">
          {FAQ.map((item) => (
            <GlassCard key={item.q}>
              <h3 className="font-medium">{item.q}</h3>
              <p className="mt-1 text-sm text-muted-foreground">{item.a}</p>
            </GlassCard>
          ))}
        </div>
      </div>
    </div>
  );
}
