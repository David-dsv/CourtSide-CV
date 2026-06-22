import { cn } from "@/lib/utils";
import type { HTMLAttributes, ReactNode } from "react";

type Glow = "green" | "cyan" | "none";

const glowClass: Record<Glow, string> = {
  green: "glow-green",
  cyan: "glow-cyan",
  none: "",
};

export function GlassCard({
  children,
  className,
  strong = false,
  glow = "none",
  ...rest
}: {
  children?: ReactNode;
  className?: string;
  strong?: boolean;
  glow?: Glow;
} & HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        strong ? "glass-strong" : "glass",
        glow !== "none" && glowClass[glow],
        "p-5",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}
