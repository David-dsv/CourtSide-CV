import { NextResponse, type NextRequest } from "next/server";

const SESSION_COOKIE = "cs_session";
const PUBLIC_PATHS = ["/", "/landing", "/pricing", "/login", "/signup"];

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;
  const session = req.cookies.get(SESSION_COOKIE)?.value;

  const isPublic = PUBLIC_PATHS.some((p) => pathname === p || pathname.startsWith(`${p}/`));
  const isApi = pathname.startsWith("/api");

  // protect everything that isn't public marketing/auth/api
  if (!isPublic && !isApi && !session) {
    const loginUrl = req.nextUrl.clone();
    loginUrl.pathname = "/login";
    loginUrl.searchParams.set("from", pathname);
    return NextResponse.redirect(loginUrl);
  }

  // if already logged in, don't show auth pages
  if (session && (pathname === "/login" || pathname === "/signup")) {
    const projectsUrl = req.nextUrl.clone();
    projectsUrl.pathname = "/projects";
    return NextResponse.redirect(projectsUrl);
  }

  return NextResponse.next();
}

export const config = {
  // skip auth for Next internals + static public assets (images, video, fonts) so
  // landing-page media (e.g. /demo/*.mp4) loads without a login redirect.
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|avif|ico|mp4|webm|mov|woff2?|ttf)$).*)",
  ],
};
