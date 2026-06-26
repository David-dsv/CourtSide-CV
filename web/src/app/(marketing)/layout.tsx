import Link from "next/link";
import { MarketingNav } from "@/components/layout/marketing-nav";
import { Logo } from "@/components/core/logo";

export default function MarketingLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col">
      <MarketingNav />
      <main className="flex-1">{children}</main>
      <footer className="border-t border-foreground/6 bg-background/60">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-4 py-10 sm:flex-row sm:px-6">
          <Link href="/" className="transition-opacity hover:opacity-80">
            <Logo />
          </Link>
          <p className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
            Local-first · zéro cloud · © 2026
          </p>
        </div>
      </footer>
    </div>
  );
}
