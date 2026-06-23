/**
 * Mock auth — session cookie only. Replaceable by NextAuth/JWT later without
 * touching the layout/middleware contract (getSession / requireSession).
 */
import { cookies } from "next/headers";
import { redirect } from "next/navigation";

const SESSION_COOKIE = "cs_session";

export interface Session {
  email: string;
  name: string;
}

export async function getSession(): Promise<Session | null> {
  const store = await cookies();
  const raw = store.get(SESSION_COOKIE)?.value;
  if (!raw) return null;
  try {
    const decoded = decodeURIComponent(raw);
    const [email, name] = decoded.split("|");
    if (!email) return null;
    return { email, name: name || email.split("@")[0] };
  } catch {
    return null;
  }
}

export async function requireSession(): Promise<Session> {
  const s = await getSession();
  if (!s) redirect("/login");
  return s;
}

export async function setSession(email: string, name?: string): Promise<void> {
  const store = await cookies();
  const value = `${email}|${name || email.split("@")[0]}`;
  store.set(SESSION_COOKIE, encodeURIComponent(value), {
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    maxAge: 60 * 60 * 24 * 7, // 7 days
  });
}

export async function clearSession(): Promise<void> {
  const store = await cookies();
  store.delete(SESSION_COOKIE);
}
