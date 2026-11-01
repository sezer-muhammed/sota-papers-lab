import { cn, text } from "@spl/ui";
import { SectionTitle } from "./section-title";

type Paper = { name: string; venue: string; url: string };

/* What each part of the app took from the literature. No paper's model or
   weights run in the app; these findings shaped its design. Sources: the
   28-paper review behind raw-retouch (research_notes/Automatic RAW photo
   retouching). */
const ROWS: { part: string; took: string; papers: Paper[] }[] = [
  {
    part: "Search each photo, don't predict",
    took: "Generic aesthetic scorers rank edits of one photo near chance, so hard limits and a constrained objective decide. Frontier models over-edit when they pick slider values, so Claude proposes looks and the GPU search sets the numbers.",
    papers: [
      { name: "PALATE", venue: "arXiv 2026", url: "https://arxiv.org/abs/2608.18622" },
      { name: "RetouchIQ", venue: "CVPR 2026", url: "https://arxiv.org/abs/2602.17558" },
      { name: "IEA", venue: "CVPR 2026 Findings", url: "https://arxiv.org/abs/2606.08016" },
    ],
  },
  {
    part: "CMA-ES over the sliders",
    took: "Black-box search over image-pipeline parameters, per photo, with no training set.",
    papers: [
      { name: "GCRL / RLPixTuner", venue: "NeurIPS 2024", url: "https://github.com/OpenImagingLab/RLPixTuner" },
      { name: "Bayesian Nights", venue: "Pattern Recognition 2025", url: "https://github.com/TheZino/Bayesian-pipeline-optimization" },
      { name: "CMA-ES tutorial (Hansen)", venue: "arXiv 2016", url: "https://arxiv.org/abs/1604.00772" },
    ],
  },
  {
    part: "Our own GPU renderer",
    took: "How Camera Raw orders its steps, white-box filters on linear RAW data, Adobe's DNG colour math and RCD demosaicing.",
    papers: [
      {
        name: "Neural Photo-Finishing",
        venue: "TOG 2022",
        url: "https://light.princeton.edu/wp-content/uploads/2022/11/Neural_Photo-Finishing.pdf",
      },
      { name: "Exposure", venue: "TOG 2018", url: "https://arxiv.org/abs/1709.09602" },
      {
        name: "Adobe DNG SDK",
        venue: "reference code",
        url: "https://android.googlesource.com/platform/external/dng_sdk/+/refs/heads/main/source/dng_render.cpp",
      },
      { name: "RCD demosaicing", venue: "Sanz Rodríguez", url: "https://github.com/LuisSR/RCD-Demosaicing" },
    ],
  },
  {
    part: "Expert priors",
    took: "Search bounds and per-slider priors measured from 33,483 expert Camera Raw edits.",
    papers: [
      {
        name: "PPR10K",
        venue: "CVPR 2021",
        url: "https://openaccess.thecvf.com/content/CVPR2021/papers/Liang_PPR10K_A_Large-Scale_Portrait_Photo_Retouching_Dataset_With_Human-Region_Mask_CVPR_2021_paper.pdf",
      },
    ],
  },
  {
    part: "Claude as analyst and planner",
    took: "Give the model the histogram, not just the picture; Lightroom Auto is the bar to beat; keep the unedited photo among the candidates.",
    papers: [
      { name: "PhotoArtAgent", venue: "CVPR 2026 Findings", url: "https://arxiv.org/abs/2505.23130" },
      { name: "RetouchLLM", venue: "arXiv 2025", url: "https://arxiv.org/abs/2510.08054" },
    ],
  },
  {
    part: "Claude as blind judge",
    took: "Pairwise choices under neutral labels, asked in both orders. The judge never steers the search, because a fixed judge gets gamed.",
    papers: [
      { name: "JarvisEvo", venue: "CVPR 2026", url: "https://arxiv.org/abs/2511.23002" },
      {
        name: "NTIRE 2026 Retouching Transfer report",
        venue: "CVPRW 2026",
        url: "https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Elezabi_Photography_Retouching_Transfer_NTIRE_2026_Challenge_Report_CVPRW_2026_paper.pdf",
      },
    ],
  },
];

export function Research() {
  return (
    <section aria-labelledby="research-title">
      <SectionTitle id="research-title" eyebrow="Built on research" title="What we took from which paper">
        From a review of 28 retouching papers. No paper's model or weights run in the app: their findings shaped
        it.
      </SectionTitle>

      <div className="mt-8">
        <div
          aria-hidden="true"
          className="hidden grid-cols-[220px_minmax(0,1fr)_300px] gap-x-8 pb-2 md:grid"
        >
          <p className={text.eyebrowSm}>In the app</p>
          <p className={text.eyebrowSm}>What we took</p>
          <p className={text.eyebrowSm}>Papers</p>
        </div>
        <ul>
          {ROWS.map((row) => (
            <li
              key={row.part}
              className="grid grid-cols-1 gap-x-8 gap-y-1.5 border-t border-[var(--ds-gray-alpha-400)] py-4 md:grid-cols-[220px_minmax(0,1fr)_300px]"
            >
              <h3 className="text-[15px] font-medium leading-6 text-[var(--ds-gray-1000)]">{row.part}</h3>
              <p className={text.small}>{row.took}</p>
              <ul aria-label={`Papers behind ${row.part.toLowerCase()}`} className="space-y-0.5 md:pt-0.5">
                {row.papers.map((paper) => (
                  <li key={paper.name} className="text-[13px] leading-5">
                    <a className={text.link} href={paper.url} rel="noreferrer" target="_blank">
                      {paper.name}
                    </a>
                    <span className={cn(text.meta, "ml-2")}>{paper.venue}</span>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
        <p className={cn(text.caption, "max-w-3xl border-t border-[var(--ds-gray-alpha-400)] pt-4")}>
          <span className="font-medium text-[var(--ds-gray-1000)]">Not used: </span>
          pixel-output diffusion retouchers such as PerTouch and SmartPhotoCrafter, because they return pixels
          rather than slider values, so nothing could go into an XMP; and closed work such as HA-ISP (closed access)
          and RetouchAgent (code removed), which we could not check.
        </p>
      </div>
    </section>
  );
}
