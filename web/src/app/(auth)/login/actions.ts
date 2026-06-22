"use server";

import { setSession } from "@/lib/mock/auth";

export async function loginAction(email: string, password: string): Promise<{ ok: true } | { ok: false; error: string }> {
  // mock auth: accept anything non-empty (real backend validates credentials)
  void password;
  if (!email || !email.includes("@")) {
    return { ok: false, error: "Email invalide" };
  }
  await setSession(email);
  return { ok: true };
}
