export const appName = 'Standard ASR';
export const docsRoute = '/docs';

export const gitConfig = {
  user: 'standard-voice',
  repo: 'standard_asr',
  branch: 'main',
};

export const githubUrl = `https://github.com/${gitConfig.user}/${gitConfig.repo}`;

// Absolute site origin plus base path, for metadata (Open Graph URLs).
export const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL ??
  `https://standard-voice.github.io${process.env.NEXT_PUBLIC_BASE_PATH ?? ''}`;

export const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
