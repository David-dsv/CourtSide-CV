"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";
import type { ComponentProps } from "react";

/**
 * App-wide theme provider (next-themes). Toggles a `.dark` class on <html>;
 * the light "Chalk" palette lives in :root, the dark "Broadcast Clay" palette
 * in .dark (see globals.css). Light is the default; the choice is persisted to
 * localStorage and there is no system-preference fallback (explicit brand themes).
 */
export function ThemeProvider({ children, ...props }: ComponentProps<typeof NextThemesProvider>) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="light"
      enableSystem={false}
      disableTransitionOnChange
      {...props}
    >
      {children}
    </NextThemesProvider>
  );
}
