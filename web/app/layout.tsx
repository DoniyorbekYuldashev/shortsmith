import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: "Shortsmith",
  description: "One line of text in, a ~30-second vertical video out.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="top">
          <Link href="/" className="brand">Shortsmith</Link>
          <nav>
            <Link href="/">New video</Link>
            <Link href="/runs">Runs</Link>
          </nav>
          <span className="note muted small">local only · mock media · billed $0 on the Gemini free tier</span>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
