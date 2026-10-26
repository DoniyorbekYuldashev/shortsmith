import type { NextConfig } from "next";

// The page talks to the API directly (http://127.0.0.1:8010, CORS allows this origin), not through a
// Next.js rewrite: a dev-server proxy can buffer a Server-Sent Events stream, and then nothing is live.
const nextConfig: NextConfig = {
  reactStrictMode: true,
  agentRules: false, // Next 16.4's `next dev` otherwise writes an AGENTS.md into web/ (Phase 8 build notes, E-list)
};

export default nextConfig;
