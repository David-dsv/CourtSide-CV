"use client";

import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";

/**
 * Light/dark toggle. Mount-guarded to avoid a hydration mismatch on the icon
 * (the server can't know the persisted theme). Flips between the "Chalk" light
 * theme and the "Broadcast Clay" dark theme.
 */
export function ThemeToggle({ className }: { className?: string }) {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // One-time mount flag: the server can't know the persisted theme, so we render
  // a neutral placeholder until the client has mounted (the documented next-themes
  // hydration-guard). This is an intentional mount-only setState, not a sync loop.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => setMounted(true), []);

  const isDark = resolvedTheme === "dark";

  return (
    <Button
      variant="ghost"
      size="icon"
      className={className}
      aria-label={isDark ? "Passer en thème clair" : "Passer en thème sombre"}
      title={isDark ? "Thème clair" : "Thème sombre"}
      onClick={() => setTheme(isDark ? "light" : "dark")}
    >
      {/* Render a stable placeholder until mounted to keep SSR/CSR markup identical. */}
      {!mounted ? (
        <Sun className="h-4 w-4 opacity-0" />
      ) : isDark ? (
        <Sun className="h-4 w-4" />
      ) : (
        <Moon className="h-4 w-4" />
      )}
    </Button>
  );
}
