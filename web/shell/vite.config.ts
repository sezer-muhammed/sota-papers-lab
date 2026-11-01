import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In dev the shell runs on your laptop and proxies /v1 to a GPU host:
//   SPL_API=http://gpu-box:7070 pnpm dev
const api = process.env.SPL_API ?? "http://127.0.0.1:7070";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { "/v1": { target: api, changeOrigin: true } },
  },
});
