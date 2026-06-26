"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  BarChart3,
  Swords,
  Target,
  CircleDot,
  Map,
  Users,
  ClipboardList,
  LineChart,
} from "lucide-react";

const NAV = [
  { href: "overview", label: "Vue d'ensemble", icon: LayoutDashboard },
  { href: "rallies", label: "Échanges", icon: Swords },
  { href: "stats", label: "Statistiques", icon: BarChart3 },
  { href: "shots", label: "Frappes", icon: Target },
  { href: "bounces", label: "Rebonds", icon: CircleDot },
  { href: "minimap", label: "Court radar", icon: Map },
  { href: "player-compare", label: "Joueurs", icon: Users },
  { href: "coach-plan", label: "Plan coach", icon: ClipboardList },
  { href: "profile", label: "Profil & tendances", icon: LineChart },
];

export function ProjectSidebar({ projectId }: { projectId: string }) {
  const pathname = usePathname();
  const base = `/projects/${projectId}`;

  return (
    <nav className="flex flex-col gap-1">
      {NAV.map((item) => {
        const href = `${base}/${item.href}`;
        const active = pathname === href || pathname.startsWith(`${href}/`);
        const Icon = item.icon;
        return (
          <Link
            key={item.href}
            href={href}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
              active
                ? "glass-strong text-foreground"
                : "text-muted-foreground hover:bg-foreground/5 hover:text-foreground",
            )}
          >
            <Icon className={cn("h-4 w-4", active && "text-court-green")} />
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
