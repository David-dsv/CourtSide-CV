import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Logo } from "@/components/core/logo";

export function MarketingNav() {
  return (
    <header className="sticky top-0 z-40 border-b border-white/6 bg-coal-950/70 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
        <Link href="/landing" className="transition-opacity hover:opacity-80">
          <Logo />
        </Link>
        <nav className="hidden items-center gap-8 font-mono text-xs uppercase tracking-widest text-muted-foreground md:flex">
          <Link href="/landing#pipeline" className="transition-colors hover:text-ball">
            Pipeline
          </Link>
          <Link href="/landing#demo" className="transition-colors hover:text-ball">
            Démo
          </Link>
          <Link href="/pricing" className="transition-colors hover:text-ball">
            Tarifs
          </Link>
        </nav>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" render={<Link href="/login" />} className="hidden sm:inline-flex">
            Se connecter
          </Button>
          <Button size="sm" render={<Link href="/signup" />} className="bg-ball text-coal-950 hover:bg-ball/90">
            Analyser ma vidéo
          </Button>
        </div>
      </div>
    </header>
  );
}
