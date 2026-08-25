import { createMDX } from 'fumadocs-mdx/next';
import path from 'node:path';

const withMDX = createMDX();

// The site is a fully static export: no Node server, no API routes at
// runtime. NEXT_PUBLIC_BASE_PATH carries the GitHub Pages project prefix
// ("/standard_asr"); an empty value builds for a root domain.
/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  reactStrictMode: true,
  trailingSlash: true,
  basePath: process.env.NEXT_PUBLIC_BASE_PATH ?? '',
  images: { unoptimized: true },
  turbopack: {
    // Content lives in the repository's docs/ tree, outside this app
    // directory, so the module graph root must be the repository root.
    root: path.resolve(process.cwd(), '../..'),
  },
};

export default withMDX(config);
