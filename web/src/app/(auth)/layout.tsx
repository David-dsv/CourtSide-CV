import Link from "next/link";
import { Logo } from "@/components/core/logo";

export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative flex min-h-screen flex-col items-center justify-center px-4 py-12">
      <div aria-hidden className="pointer-events-none absolute inset-0 court-grid opacity-40" />
      <Link href="/" className="relative mb-8 transition-opacity hover:opacity-80">
        <Logo className="scale-110" />
      </Link>
      <div className="relative w-full max-w-sm">{children}</div>
    </div>
  );
}
