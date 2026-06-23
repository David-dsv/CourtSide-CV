import Link from "next/link";
import { Plus } from "lucide-react";
import { requireSession } from "@/lib/mock/auth";
import { UserMenu } from "@/components/layout/user-menu";
import { Button } from "@/components/ui/button";
import { Logo } from "@/components/core/logo";

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const session = await requireSession();

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b border-white/6 bg-coal-950/70 backdrop-blur-xl">
        <div className="flex h-14 items-center justify-between px-4 sm:px-6">
          <Link href="/landing" className="transition-opacity hover:opacity-80">
            <Logo />
          </Link>
          <div className="flex items-center gap-2">
            <Button size="sm" render={<Link href="/projects/new" />} className="bg-ball text-coal-950 hover:bg-ball/90">
              <Plus className="mr-1 h-4 w-4" /> Nouveau match
            </Button>
            <UserMenu email={session.email} name={session.name} />
          </div>
        </div>
      </header>
      <main className="flex-1">{children}</main>
    </div>
  );
}
