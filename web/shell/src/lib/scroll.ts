import { useLayoutEffect, useRef, type MouseEvent } from "react";

/* In-page jumps and focus moves. Never use bare "#id" anchors: the hash
   router would read them as routes. */

export function prefersReducedMotion() {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

export function scrollToId(id: string) {
  document
    .getElementById(id)
    ?.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "instant" : "smooth" });
}

/** Focuses `el` without scrolling. An element that isn't focusable gets tabindex="-1" first. */
function focusWithoutScroll(el: HTMLElement) {
  if (!el.hasAttribute("tabindex")) el.setAttribute("tabindex", "-1");
  el.focus({ preventScroll: true });
}

/** Scrolls to a section and moves focus there, so keyboard and screen-reader
    users land where the page went. */
export function jumpTo(id: string) {
  const target = document.getElementById(id);
  if (!target) return;
  scrollToId(id);
  focusWithoutScroll(target);
}

/** For links to #/host: a click while already there fires no hashchange, so jump by hand. */
export function sameHashScroll(event: MouseEvent<HTMLAnchorElement>) {
  if (window.location.hash !== "#/host") return;
  event.preventDefault();
  jumpTo("host");
}

/** Brings the Apps panel's token form into view and focuses its input. */
export function focusTokenInput() {
  scrollToId("apps");
  requestAnimationFrame(() =>
    document.getElementById("spl-token-input")?.focus({ preventScroll: true }),
  );
}

/** Ref for a control that goes away once its action works (a Retry). If focus
    is inside it when it unmounts, focus moves to the closest `within` ancestor
    instead of dropping to <body>. Layout cleanups run before React removes the
    node, so the check still sees it focused. */
export function useFocusHandoff<T extends HTMLElement>(within = "section") {
  const ref = useRef<T>(null);

  useLayoutEffect(() => {
    const node = ref.current;
    return () => {
      if (!node?.contains(document.activeElement)) return;
      const target = node.parentElement?.closest<HTMLElement>(within);
      if (target) focusWithoutScroll(target);
    };
  }, [within]);

  return ref;
}
