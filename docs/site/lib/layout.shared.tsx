import Image from 'next/image';
import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import icon from '@/app/icon.png';
import { appName, githubUrl } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <span className="inline-flex items-center gap-2 font-mono text-[15px] font-bold tracking-tight">
          <Image src={icon} alt="" width={20} height={20} className="rounded-[5px]" />
          {appName}
        </span>
      ),
    },
    githubUrl,
  };
}
