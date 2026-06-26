"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { GlassCard } from "@/components/core/glass-card";
import { signupAction } from "./actions";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";

export default function SignupPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    const res = await signupAction(email, name || email.split("@")[0], password);
    if (res.ok) {
      toast.success("Compte créé");
      router.push("/projects");
      router.refresh();
    } else {
      setLoading(false);
      toast.error(res.error);
    }
  }

  return (
    <GlassCard strong className="w-full max-w-sm">
      <div className="mb-6 text-center">
        <h1 className="text-xl font-semibold">Créer un compte</h1>
        <p className="mt-1 text-sm text-muted-foreground">Analysez votre premier match gratuitement</p>
      </div>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <div className="flex flex-col gap-2">
          <Label htmlFor="name">Nom</Label>
          <Input id="name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Votre nom" />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="email">Email</Label>
          <Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="password">Mot de passe</Label>
          <Input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required autoComplete="new-password" />
        </div>
        <Button type="submit" disabled={loading} className="bg-ball text-primary-foreground hover:bg-ball/90">
          {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Créer mon compte
        </Button>
      </form>
      <p className="mt-6 text-center text-sm text-muted-foreground">
        Déjà un compte ?{" "}
        <Link href="/login" className="text-court-green hover:underline">
          Se connecter
        </Link>
      </p>
    </GlassCard>
  );
}
