/*
 * Central type scale for the whole site. Every page composes these
 * constants instead of hardcoding font sizes, so adjusting the scale
 * is a one-file change.
 */
export const text = {
  /* mono uppercase labels above titles */
  eyebrow: "font-mono text-[13px] uppercase text-[var(--ds-gray-700)]",
  eyebrowSm: "font-mono text-[12px] uppercase text-[var(--ds-gray-700)]",
  /* h1 on the landing hero */
  display: "text-[42px] font-semibold leading-[1.03] sm:text-[58px]",
  /* h1 on inner pages (checker, pricing, blog) */
  pageTitle: "text-[34px] font-semibold leading-[1.08] sm:text-[42px]",
  /* h2 inside CTA bands */
  bandTitle: "text-[28px] font-semibold leading-9",
  /* intro paragraph under a page title */
  lead: "text-[17px] leading-8 text-[var(--ds-gray-900)]",
  /* emphasized body copy */
  body: "text-[15px] leading-6 text-[var(--ds-gray-900)]",
  /* default supporting copy */
  small: "text-[14px] leading-6 text-[var(--ds-gray-900)]",
  /* mono metadata (dates, counters, read time) */
  meta: "font-mono text-[12px] text-[var(--ds-gray-700)]",
} as const;
