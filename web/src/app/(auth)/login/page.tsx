"use client";

import { useState, Suspense } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { GlassCard } from "@/components/core/glass-card";
import { loginAction } from "./actions";
import { Loader2, Activity } from "lucide-react";
import { toast } from "sonner";

export default function LoginPage() {
  return (
    <Suspense fallback={<GlassCard strong className="w-full max-w-sm p-8" />}>
      <LoginForm />
    </Suspense>
  );
}

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const from = params.get("from") || "/projects";
  const [email, setEmail] = useState("demo@courtside.app");
  const [password, setPassword] = useState("demo");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    const res = await loginAction(email, password);
    if (res.ok) {
      toast.success("Connecté");
      router.push(from);
      router.refresh();
    } else {
      setLoading(false);
      toast.error(res.error);
    }
  }

  return (
    <GlassCard strong className="w-full max-w-sm">
      <div className="mb-6 text-center">
        <h1 className="text-xl font-semibold">Bon retour</h1>
        <p className="mt-1 text-sm text-muted-foreground">Connectez-vous à votre tableau de bord</p>
      </div>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <div className="flex flex-col gap-2">
          <Label htmlFor="email">Email</Label>
          <Input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            autoComplete="email"
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="password">Mot de passe</Label>
          <Input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            autoComplete="current-password"
          />
        </div>
        <Button type="submit" disabled={loading} className="bg-ball text-coal-950 hover:bg-ball/90">
          {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Se connecter
        </Button>
      </form>
      <div className="my-4 flex items-center gap-3 text-xs text-muted-foreground">
        <div className="h-px flex-1 bg-white/10" />
        DÉMO
        <div className="h-px flex-1 bg-white/10" />
      </div>
      <Button
        variant="outline"
        className="w-full"
        onClick={() => {
          setEmail("demo@courtside.app");
          setPassword("demo");
        }}
      >
        <Activity className="mr-2 h-4 w-4 text-court-green" />
        Pré-remplir le compte démo
      </Button>
      <p className="mt-6 text-center text-sm text-muted-foreground">
        Pas de compte ?{" "}
        <Link href="/signup" className="text-court-green hover:underline">
          Créer un compte
        </Link>
      </p>
    </GlassCard>
  );
}
