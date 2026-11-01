import { cn } from "../lib/cn";

/* Brand symbol: a stack of papers with the newest (top) sheet in blue, the
   one action color of the system. Other sheets use currentColor so dark mode
   just works. */
export function BrandMark({ className }: { className?: string }) {
  return (
    <svg aria-hidden="true" className={className} fill="none" viewBox="0 0 64 64">
      <rect x="6" y="30" width="44" height="28" rx="4" fill="currentColor" opacity="0.35" />
      <rect x="10" y="20" width="44" height="28" rx="4" fill="currentColor" opacity="0.65" />
      <rect x="14" y="8" width="44" height="30" rx="4" fill="var(--ds-blue-700)" />
      <path d="M22 17h20M22 24h28M22 31h14" stroke="white" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

/* Full lockup: mark + bold SOTA / light PAPERS LAB, same rhythm as ATSGrade. */
export function BrandLogo({ className }: { className?: string }) {
  return (
    <span className={cn("flex min-w-0 items-center gap-2.5", className)}>
      <BrandMark className="h-7 w-7 shrink-0" />
      <span className="truncate text-[17px] leading-none tracking-tight">
        <span className="font-extrabold">SOTA</span>
        <span className="ml-1.5 font-light tracking-wide">PAPERS LAB</span>
      </span>
    </span>
  );
}
