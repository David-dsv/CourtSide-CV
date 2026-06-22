import { NextResponse } from "next/server";
import { setSession } from "@/lib/mock/auth";

/** One-click demo entry: sets a mock session and lands on the projects grid. */
export async function GET(req: Request) {
  await setSession("demo@courtside.app", "Démo");
  return NextResponse.redirect(new URL("/projects", req.url));
}
