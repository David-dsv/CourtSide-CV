"use server";

import { setSession } from "@/lib/mock/auth";

export async function signupAction(
  email: string,
  name: string,
  _password: string,
): Promise<{ ok: true } | { ok: false; error: string }> {
  if (!email || !email.includes("@")) {
    return { ok: false, error: "Email invalide" };
  }
  if (!_password || _password.length < 1) {
    return { ok: false, error: "Mot de passe requis" };
  }
  await setSession(email, name);
  return { ok: true };
}
